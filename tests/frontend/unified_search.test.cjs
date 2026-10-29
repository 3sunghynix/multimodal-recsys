const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

// 실제 HTML을 브라우저에서 실행하되 API만 대역으로 바꾼다. DB·모델·외부 네트워크 불필요.
const html = fs.readFileSync(path.join(__dirname, '../../backend/frontend/index.html'), 'utf8');
const image = { name: 'dress.png', mimeType: 'image/png', buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jV9sAAAAASUVORK5CYII=', 'base64') };
const product = { id: 8, source: 'amazon', brand_id: 2, title: 'Black dress', category: 'dress', price: 10, image_path: null };
let browser;
before(async () => {
  browser = await chromium.launch({ headless: true, ...(process.env.MSRS_TEST_BROWSER_CHANNEL ? { channel: process.env.MSRS_TEST_BROWSER_CHANNEL } : {}) });
});
after(async () => { if (browser) await browser.close(); });

async function openDemo(t, respond = async () => ({ query_en: 'black dress', untranslated: [], items: [product] })) {
  const page = await browser.newPage();
  const requests = [];
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  t.after(async () => { await page.close(); assert.deepEqual(errors, [], 'No unhandled browser errors'); });
  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== 'https://msrs.test') return route.abort();
    if (url.pathname === '/app/') return route.fulfill({ contentType: 'text/html; charset=utf-8', body: html });
    requests.push({ url: url.pathname + url.search, method: request.method(), body: request.postData() || '', contentType: request.headers()['content-type'] });
    const result = await respond(request, requests.length);
    if (result === 'abort') return route.abort();
    return route.fulfill({ status: result.status || 200, contentType: 'application/json', body: JSON.stringify(result.body || result) });
  });
  await page.goto('https://msrs.test/app/');
  return { page, requests };
}

async function search(page) {
  await page.click('#search-submit');
  await page.waitForSelector('#status.ok');
}

test('Korean text uses multipart POST /search, response details directly and translation notice', async (t) => {
  const { page, requests } = await openDemo(t);
  await page.fill('#search-query', '검은색 원피스');
  assert.equal(await page.locator('#weight-control').isVisible(), false);
  await search(page);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, '/search');
  assert.equal(requests[0].method, 'POST');
  assert.match(requests[0].contentType, /multipart\/form-data; boundary=/);
  assert.match(requests[0].body, /name="q"\r\n\r\n검은색 원피스/);
  assert.match(requests[0].body, /name="k"\r\n\r\n20/);
  assert.doesNotMatch(requests[0].body, /name="file"/);
  assert.match(await page.locator('#translation-note').innerText(), /black dress/);
  assert.equal(await page.locator('.card .title').innerText(), 'Black dress');
});

test('image-only search allows an empty text input and retains image until explicit removal', async (t) => {
  const { page, requests } = await openDemo(t, async () => ({ query_en: null, untranslated: [], items: [product] }));
  await page.setInputFiles('#img-input', image);
  assert.equal(requests.length, 0, 'Selection waits for explicit submit');
  assert.equal(await page.locator('#weight-control').isVisible(), false);
  await search(page);
  assert.match(requests[0].body, /name="file"; filename="dress.png"/);
  assert.doesNotMatch(requests[0].body, /name="q"/);
  assert.equal(await page.locator('#translation-note').isVisible(), false);
  assert.equal(await page.locator('#image-selection').isVisible(), true);
});

test('combined search sends both inputs and selected weight including endpoints 0 and 1', async (t) => {
  const { page, requests } = await openDemo(t);
  await page.fill('#search-query', 'dress');
  await page.setInputFiles('#img-input', image);
  assert.equal(await page.locator('#weight-control').isVisible(), true);
  for (const weight of ['0', '0.3', '1']) {
    await page.locator('#text-weight').evaluate((el, value) => { el.value = value; el.dispatchEvent(new Event('input', { bubbles: true })); }, weight);
    await search(page);
    const body = requests.at(-1).body;
    assert.match(body, /name="q"\r\n\r\ndress/);
    assert.match(body, /name="file"; filename="dress.png"/);
    assert.ok(body.includes(`name="text_weight"\r\n\r\n${weight}\r\n`));
  }
  assert.match(await page.locator('#weight-label').innerText(), /텍스트 100% · 이미지 0%/);
});

test('removing image hides slider and next search does not submit stale image', async (t) => {
  const { page, requests } = await openDemo(t);
  await page.fill('#search-query', 'dress');
  await page.setInputFiles('#img-input', image);
  await page.click('#clear-image');
  assert.equal(await page.locator('#weight-control').isVisible(), false);
  assert.equal(await page.locator('#image-selection').isVisible(), false);
  await search(page);
  assert.doesNotMatch(requests[0].body, /name="file"/);
});

test('untranslated words are shown and removed after a subsequent fully translated query', async (t) => {
  const { page } = await openDemo(t, async (_, n) => ({ query_en: 'dress', untranslated: n === 1 ? ['모르는단어'] : [], items: [product] }));
  await page.fill('#search-query', '모르는단어 원피스');
  await search(page);
  assert.match(await page.locator('.translation-note .warning').innerText(), /번역 못한 단어: 모르는단어/);
  await page.fill('#search-query', '원피스');
  await search(page);
  assert.equal(await page.locator('.translation-note .warning').count(), 0);
});

test('empty input shows guidance without an API request', async (t) => {
  const { page, requests } = await openDemo(t);
  await page.fill('#search-query', '   ');
  await page.click('#search-submit');
  await page.waitForSelector('#status.error');
  assert.equal(requests.length, 0);
});

test('invalid image is rejected locally', async (t) => {
  const { page, requests } = await openDemo(t);
  await page.setInputFiles('#img-input', { name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('text') });
  assert.equal(await page.locator('#status.error').count(), 1);
  assert.equal(await page.locator('#image-selection').isVisible(), false);
  assert.equal(requests.length, 0);
});

test('HTTP errors clear previous results and a retry can recover', async (t) => {
  const { page } = await openDemo(t, async (_, n) => n === 2 ? { status: 503, body: { detail: 'PCA 파일 미설치' } } : { query_en: 'dress', untranslated: [], items: [product] });
  await page.fill('#search-query', 'dress');
  await search(page);
  await page.click('#search-submit');
  await page.waitForSelector('#status.error');
  assert.match(await page.locator('#status').innerText(), /HTTP 503.*PCA/);
  assert.equal(await page.locator('#result-section').isVisible(), false);
  await search(page);
  assert.equal(await page.locator('.card').count(), 1);
});

test('network failure shows useful message instead of an unhandled rejection', async (t) => {
  const { page } = await openDemo(t, async () => 'abort');
  await page.fill('#search-query', 'dress');
  await page.click('#search-submit');
  await page.waitForSelector('#status.error');
  assert.match(await page.locator('#status').innerText(), /서버에 연결하지 못했습니다/);
});

test('catalogue and translation markup are displayed literally without executing', async (t) => {
  const payload = '<img src=x onerror="window.__injected=true">';
  const { page } = await openDemo(t, async () => ({ query_en: payload, untranslated: [payload], items: [{ ...product, title: payload, source: payload }] }));
  await page.fill('#search-query', payload);
  await search(page);
  assert.equal(await page.locator('.title').innerText(), payload);
  assert.equal(await page.locator('.title img, .src-badge img, #translation-note img, #status img').count(), 0);
  assert.equal(await page.evaluate(() => window.__injected), undefined);
});

test('late response does not replace newer results', async (t) => {
  let releaseOld;
  const gate = new Promise((resolve) => { releaseOld = resolve; });
  const { page, requests } = await openDemo(t, async (_, n) => {
    if (n === 1) await gate;
    return { query_en: n === 1 ? 'old' : 'new', untranslated: [], items: [{ ...product, title: n === 1 ? 'Old result' : 'New result' }] };
  });
  await page.fill('#search-query', 'old');
  await page.click('#search-submit');
  await page.waitForFunction(() => document.querySelector('#result-section').getAttribute('aria-busy') === 'true');
  // Wait until first request reached our API double before sending the next query.
  while (requests.length < 1) await new Promise((resolve) => setImmediate(resolve));
  await page.fill('#search-query', 'new');
  await search(page);
  releaseOld();
  assert.equal(await page.locator('.title').innerText(), 'New result');
});

test('recommendation still fetches details and clears previous translation notice', async (t) => {
  const { page, requests } = await openDemo(t, async (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith('/recommend/')) return { model: 'popularity', variant: 'A', items: [8] };
    if (url.pathname === '/items') return [product];
    return { query_en: 'dress', untranslated: [], items: [product] };
  });
  await page.fill('#search-query', '원피스');
  await search(page);
  await page.click('#btn-recommend-top');
  await page.waitForFunction(() => document.querySelector('#result-title').textContent.includes('고객 #1'));
  assert.deepEqual(requests.map((r) => r.url), ['/search', '/recommend/1?k=20', '/items?ids=8']);
  assert.equal(await page.locator('#translation-note').isVisible(), false);
  assert.match(await page.locator('#result-meta').innerText(), /A\/B: A/);
});

test('combined controls remain visible without horizontal overflow on a small screen', async (t) => {
  const { page } = await openDemo(t);
  await page.fill('#search-query', '검은색 원피스');
  await page.setInputFiles('#img-input', image);
  await search(page);
  if (process.env.MSRS_PREVIEW_DIR) {
    fs.mkdirSync(process.env.MSRS_PREVIEW_DIR, { recursive: true });
    await page.screenshot({ path: path.join(process.env.MSRS_PREVIEW_DIR, 'desktop_mock.png'), fullPage: true });
  }
  await page.setViewportSize({ width: 375, height: 812 });
  assert.equal(await page.locator('#weight-control').isVisible(), true);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  const form = await page.locator('#form-search').boundingBox();
  const header = await page.locator('.header').boundingBox();
  assert.ok(form.y >= header.y + header.height, 'Fixed header does not cover search controls');
  if (process.env.MSRS_PREVIEW_DIR) {
    await page.screenshot({ path: path.join(process.env.MSRS_PREVIEW_DIR, 'mobile_mock.png'), fullPage: true });
  }
});
