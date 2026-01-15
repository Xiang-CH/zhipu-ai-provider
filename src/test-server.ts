import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

export type TestServerResponse =
  | {
      type: 'json-value';
      headers?: Record<string, string>;
      body: unknown;
    }
  | {
      type: 'stream-chunks';
      headers?: Record<string, string>;
      chunks: string[];
    };

type TestServerUrl = {
  response: TestServerResponse;
  calls: Array<{
    requestBody: string;
    requestBodyJson: unknown;
    requestHeaders: Record<string, string>;
  }>;
};

type ServerCall = {
  requestBody: string;
  requestBodyJson: unknown;
  requestHeaders: Record<string, string>;
};

export type TestServer = {
  urls: Record<string, TestServerUrl>;
  calls: ServerCall[];
  close: () => void;
};

export function createTestServer(
  urls: Record<string, Omit<TestServerUrl, 'calls'>>,
): TestServer {
  const serverUrls: Record<string, TestServerUrl> = {};
  const allCalls: ServerCall[] = [];

  for (const [url, value] of Object.entries(urls)) {
    serverUrls[url] = {
      ...value,
      calls: [],
    };
  }

  const handlers = Object.entries(serverUrls).map(([url, urlConfig]) =>
    http.post(url, async ({ request }) => {
      const requestBody = await request.text();
      const requestHeaders: Record<string, string> = {};
      request.headers.forEach((value, key) => {
        requestHeaders[key] = value;
      });

      let requestBodyJson: unknown;
      try {
        requestBodyJson = JSON.parse(requestBody);
      } catch (e) {
        console.error('Failed to parse request body:', requestBody);
        throw e;
      }

      const call: ServerCall = {
        requestBody,
        requestHeaders,
        requestBodyJson,
      };

      urlConfig.calls.push(call);
      allCalls.push(call);

      const response = urlConfig.response;

      if (response.type === 'json-value') {
        return HttpResponse.json(response.body, {
          headers: response.headers,
        });
      }

      if (response.type === 'stream-chunks') {
        return new Response(response.chunks.join(''), {
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
            ...response.headers,
          },
        });
      }

      return new Response(null, { status: 500 });
    }),
  );

  const server = setupServer(...handlers);
  server.listen();

  return {
    urls: serverUrls,
    calls: allCalls,
    close: () => server.close(),
  };
}
