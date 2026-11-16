"""Run a read-only public demo tunnel until an absolute UTC deadline."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent.parent

def stop(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expires-at',required=True)
    parser.add_argument('--state-file',required=True)
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.expires_at.replace('Z','+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('expires-at must include a time zone')
    deadline=deadline.astimezone(timezone.utc)
    if deadline<=datetime.now(timezone.utc):
        raise ValueError('The expiry time must be in the future')
    executable=ROOT/'tools'/'cloudflared.exe'
    if not executable.is_file():
        raise FileNotFoundError(executable)
    state_path=Path(args.state_file).resolve()
    state_path.parent.mkdir(parents=True,exist_ok=True)
    gateway=tunnel=None
    state={'started_at':datetime.now(timezone.utc).isoformat(),'expires_at':deadline.isoformat(),
           'gateway':'http://127.0.0.1:8766','upstream':'http://127.0.0.1:8765','read_only':True,
           'manager_pid':__import__('os').getpid(),'status':'starting'}
    def save():
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        gateway=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/'public_gateway.py'),
            '--port','8766','--upstream','http://127.0.0.1:8765','--expires-at',deadline.isoformat()],
            cwd=str(ROOT),stdin=subprocess.DEVNULL)
        state['gateway_pid']=gateway.pid
        save()
        # Readiness checks are bounded; no public tunnel starts before the guard is live.
        ready=False
        for _ in range(45):
            if gateway.poll() is not None:
                raise RuntimeError(f'Gateway exited with {gateway.returncode}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8766/api/health',timeout=2) as response:
                    ready=response.status==200
                if ready:
                    break
            except (OSError,ValueError):
                pass
            time.sleep(1)
        if not ready:
            raise RuntimeError('Read-only gateway did not become ready')
        tunnel=subprocess.Popen([str(executable),'tunnel','--url','http://127.0.0.1:8766',
            '--no-autoupdate','--protocol','http2','--edge-ip-version','4'],cwd=str(ROOT),stdin=subprocess.DEVNULL)
        state.update(tunnel_pid=tunnel.pid,status='running')
        save()
        print('PUBLIC_DEMO_EXPIRY='+deadline.isoformat(),flush=True)
        while True:
            remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
            if remaining<=0:
                state['status']='expired'
                print('24-hour public access expired; closing tunnel and gateway.',flush=True)
                break
            if gateway.poll() is not None:
                raise RuntimeError(f'Gateway exited with {gateway.returncode}')
            try:
                tunnel.wait(timeout=min(60,remaining))
            except subprocess.TimeoutExpired:
                continue
            raise RuntimeError(f'Cloudflare tunnel exited with {tunnel.returncode}')
    except Exception as error:
        state.update(status='failed',error=str(error))
        raise
    finally:
        stop(tunnel)
        stop(gateway)
        state['stopped_at']=datetime.now(timezone.utc).isoformat()
        save()

if __name__=='__main__':
    main()
