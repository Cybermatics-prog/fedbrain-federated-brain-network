# SPDX-License-Identifier: MIT
"""No model, server, network, training or cache jobs are imported/started here."""
import gzip
import unittest
from datetime import datetime, timezone

import httpx
from fastapi.testclient import TestClient

from scripts.public_gateway import READ_ONLY_MESSAGE, UPSTREAM_UNAVAILABLE, allowed_path, create_app


class PublicGatewayTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

        def handler(request):
            self.requests.append(request)
            self.assertEqual(str(request.url.copy_with(query=None)).split('/')[0:3],
                             ['http:', '', '127.0.0.1:8765'])
            path = request.url.path
            if path == '/':
                return httpx.Response(200, text='<html>single-process simulation demo</html>',
                                      headers={'content-type': 'text/html; charset=utf-8'})
            if path.endswith('.js'):
                return httpx.Response(200, content=gzip.compress(b'console.log("demo")'), headers={
                    'content-type': 'text/javascript', 'content-encoding': 'gzip',
                    'cache-control': 'public, max-age=60', 'set-cookie': 'private=secret',
                    'connection': 'keep-alive, x-private', 'x-private': 'secret',
                })
            if path.endswith('.css'):
                return httpx.Response(200, text='body { color: black; }',
                                      headers={'content-type': 'text/css'})
            if path == '/api/health-data/sites':
                return httpx.Response(200, json={'sites': [{'id': str(i)} for i in range(6)]})
            if path == '/api/subjects/invalid':
                return httpx.Response(409, json={'detail': 'The demo model is not ready yet'})
            if path == '/api/runs/not-found':
                return httpx.Response(404, json={'detail': 'Run not found'})
            return httpx.Response(200, json={'ok': True})

        self.app = create_app(transport=httpx.MockTransport(handler))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_root_and_six_site_api_readable(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn('text/html', page.headers['content-type'])
        self.assertIn('single-process', page.text)
        response = self.client.get('/api/health-data/sites')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['sites']), 6)
        for path in ('/api/health', '/api/status', '/api/protocol', '/api/optimization',
                     '/api/runs', '/api/runs/demo-42', '/api/runs/demo-42/federation',
                     '/api/runs/demo-42/federation/1', '/api/runs/demo-42/federation/500',
                     '/api/data-protection', '/api/health-data',
                     '/api/health-data/subjects/50957', '/api/subjects', '/api/subjects/50957'):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_writes_always_forbidden_and_never_reach_upstream(self):
        for method in ('POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'TRACE'):
            for path in ('/api/train', '/api/runs/demo/stop', '/', '/api/status'):
                with self.subTest(method=method, path=path):
                    response = self.client.request(method, path, json={'rounds': 1})
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(response.json()['detail'], READ_ONLY_MESSAGE)
        self.assertEqual(self.requests, [])

    def test_forbidden_and_unknown_paths_never_reach_upstream(self):
        paths = ('/server.py', '/web/secret.py.js', '/checkpoint.pt',
                 '/checkpoint/file.js', '/api/runs/demo/download', '/export/data.js',
                 '/api/runs/demo/federation/0', '/api/runs/demo/federation/501',
                 '/api/runs/demo/federation/1/extra', '/api/train', '/api/docs',
                 '/openapi.json', '/unknown-route', '/api/unknown.js',
                 '/%2e%2e/secret.js', '/%252e%252e/secret.js', '/web%5csecret.js',
                 '/api/subjects/%2e%2e')
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.requests, [])
        for path in ('/../file.js', '/web/../file.js', '//evil.test/file.js',
                     'http://evil.test/file.js', '/x/./file.css'):
            self.assertFalse(allowed_path(path))

    def test_expired_all_requests_return_410_without_upstream(self):
        now = datetime(2026, 11, 14, 7, 0, tzinfo=timezone.utc)
        with TestClient(create_app(
            expires_at='2026-11-14T07:00:00Z', now=lambda: now,
            transport=httpx.MockTransport(lambda r: self.fail('expired upstream request')),
        )) as client:
            for method, path in (('GET', '/'), ('HEAD', '/api/status'),
                                 ('POST', '/api/train'), ('GET', '/unknown')):
                self.assertEqual(client.request(method, path).status_code, 410)

    def test_status_and_static_headers_and_decoding(self):
        response = self.client.get('/assets/app.js')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'console.log("demo")')
        self.assertEqual(response.headers['content-type'], 'text/javascript')
        self.assertEqual(response.headers['cache-control'], 'public, max-age=60')
        self.assertEqual(int(response.headers['content-length']), len(response.content))
        for header in ('content-encoding', 'connection', 'x-private', 'set-cookie'):
            self.assertNotIn(header, response.headers)
        self.assertEqual(self.client.get('/style.css').headers['content-type'], 'text/css')
        self.assertEqual(self.client.get('/api/subjects/invalid').status_code, 409)
        self.assertEqual(self.client.get('/api/runs/not-found').status_code, 404)
        head = self.client.head('/api/status')
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b'')
        self.assertEqual(self.requests[-1].method, 'HEAD')

    def test_no_credentials_and_target_always_fixed(self):
        # The upstream deliberately sets a cookie; it must not reappear on
        # subsequent requests via httpx's internal cookie jar either.
        self.client.get('/assets/app.js')
        self.client.get('/api/status?url=http://evil.test&run_id=demo', headers={
            'cookie': 'secret=abc', 'authorization': 'Bearer secret',
            'host': 'evil.test', 'x-forwarded-host': 'evil.test',
            'proxy-authorization': 'Basic secret',
        })
        request = self.requests[-1]
        self.assertEqual(request.url.host, '127.0.0.1')
        self.assertEqual(request.url.port, 8765)
        self.assertEqual(request.url.params['run_id'], 'demo')
        for header in ('cookie', 'authorization', 'proxy-authorization', 'x-forwarded-host'):
            self.assertNotIn(header, request.headers)
        self.assertEqual(request.headers['host'], '127.0.0.1:8765')
        with self.assertRaises(ValueError):
            create_app(upstream='http://evil.test')
        with self.assertRaises(ValueError):
            create_app(expires_at='2026-11-14T07:00:00')

    def test_upstream_network_failure_returns_sanitized_502(self):
        def fail(request):
            raise httpx.ConnectError('SECRET traceback path', request=request)
        with TestClient(create_app(transport=httpx.MockTransport(fail))) as client:
            response = client.get('/api/health')
            self.assertEqual(response.status_code, 502)
            self.assertNotIn('SECRET', response.text)
            self.assertEqual(response.json()['detail'], UPSTREAM_UNAVAILABLE)


if __name__ == '__main__':
    unittest.main()
