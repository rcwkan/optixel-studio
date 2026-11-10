/**
 * Cloudflare Worker: makecade-optixel-router
 * Router & Edge Proxy for Optixel Studio (https://optixel.makecade.com)
 *
 * Responsibilities:
 * - Proxies static assets & HTML to Cloudflare Pages origin (optixel-studio.pages.dev)
 * - Single Page Application (SPA) routing fallback for deep URLs
 * - High-performance edge caching for assets (/icon.png, /assets/*)
 * - Security headers enforcement
 * - Health check endpoint (/__health)
 */

const DEFAULT_PAGES_ORIGIN = 'https://optixel-studio.pages.dev';

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const pathname = url.pathname;
    const pagesOrigin = (env.PAGES_ORIGIN || DEFAULT_PAGES_ORIGIN).replace(/\/+$/, '');

    // 1. Health check probe endpoint
    if (pathname === '/__health') {
      return new Response(
        JSON.stringify(
          {
            status: 'ok',
            service: 'makecade-optixel-router',
            targetDomain: 'optixel.makecade.com',
            pagesOrigin,
            timestamp: new Date().toISOString(),
          },
          null,
          2
        ),
        {
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
            'Cache-Control': 'no-store',
          },
        }
      );
    }

    let targetPath = pathname;

    // 2. Proxy to Cloudflare Pages deployment
    const pagesUrl = new URL(targetPath + url.search, pagesOrigin);
    const pagesHost = pagesOrigin.replace(/^https?:\/\//i, '').split('/')[0];

    const forwardHeaders = new Headers(request.headers);
    forwardHeaders.set('Host', pagesHost);
    forwardHeaders.set('X-Forwarded-Host', url.host);
    forwardHeaders.set('X-Forwarded-Proto', url.protocol.replace(':', ''));

    const proxyRequest = new Request(pagesUrl.toString(), {
      method: request.method,
      headers: forwardHeaders,
      body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : null,
      redirect: 'follow',
    });

    try {
      let response = await fetch(proxyRequest);

      // 3. SPA Fallback: If 404 and no file extension, serve index.html
      const hasExtension = /\.[a-zA-Z0-9]+$/.test(targetPath);
      if (response.status === 404 && !hasExtension) {
        const spaIndexUrl = new URL('/index.html', pagesOrigin);
        const spaRequest = new Request(spaIndexUrl.toString(), {
          method: 'GET',
          headers: forwardHeaders,
        });
        const spaResponse = await fetch(spaRequest);
        if (spaResponse.status === 200) {
          response = spaResponse;
        }
      }

      const responseHeaders = new Headers(response.headers);

      // Security headers
      responseHeaders.set('X-Content-Type-Options', 'nosniff');
      responseHeaders.set('X-Frame-Options', 'SAMEORIGIN');
      responseHeaders.set('Referrer-Policy', 'strict-origin-when-cross-origin');

      // Edge and browser caching rules
      if (pathname.startsWith('/assets/') || pathname === '/icon.png') {
        responseHeaders.set('Cache-Control', 'public, max-age=31536000, immutable');
      } else if (responseHeaders.get('Content-Type')?.includes('text/html')) {
        responseHeaders.set('Cache-Control', 'public, max-age=0, must-revalidate');
      }

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
      });
    } catch (err) {
      return new Response(`Error proxying to Optixel Studio Pages origin: ${err.message}`, {
        status: 502,
        headers: { 'Content-Type': 'text/plain' },
      });
    }
  },
};
