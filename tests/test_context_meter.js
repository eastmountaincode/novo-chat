const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const javascript = readFileSync(join(__dirname, "../novo_chat/gateway/web/app.js"), "utf8");
const source = "function renderContextMeter(result) {" + javascript
  .split("function renderContextMeter(result) {", 2)[1]
  .split("function safeSourceHref", 1)[0];

function render(timings) {
  const fill = { style: {} };
  const meter = {
    innerHTML: "previous answer's meter",
    querySelector: (selector) => {
      assert.equal(selector, ".meter > div");
      return fill;
    },
  };
  const context = vm.createContext({
    el: { contextMeter: meter },
    formatInt: (value) => Number(value).toLocaleString("en-US"),
  });
  vm.runInContext(source, context);
  context.renderContextMeter({ timings });
  return { html: meter.innerHTML, width: fill.style.width };
}

test("meter adds input and actual output, not the output allowance", () => {
  const meter = render({ prompt_eval_count: 8192, eval_count: 4096, num_ctx: 32768, max_tokens: 16384 });
  assert.match(meter.html, /12,288 \/ 32,768 tokens/);
  assert.match(meter.html, /38% full/);
  assert.match(meter.html, /8,192 input \+ 4,096 output \(including thinking\)/);
  assert.equal(meter.width, "37.5%");
});

test("a reported zero output count is valid", () => {
  const meter = render({ prompt_eval_count: 100, eval_count: 0, num_ctx: 1000 });
  assert.match(meter.html, /100 \/ 1,000 tokens/);
  assert.equal(meter.width, "10%");
});

test("legacy, absent, or invalid output usage is not shown as total context", () => {
  for (const eval_count of [undefined, null, -1, true, "100", 0.5]) {
    assert.equal(render({ prompt_eval_count: 100, eval_count, num_ctx: 1000 }).html, "");
  }
  assert.equal(render(undefined).html, "");
});

test("full context caps the bar, not the actual token count", () => {
  const meter = render({ prompt_eval_count: 900, eval_count: 200, num_ctx: 1000 });
  assert.match(meter.html, /1,100 \/ 1,000 tokens/);
  assert.match(meter.html, /aria-valuenow="100"/);
  assert.equal(meter.width, "100%");
});

test("each selected answer retains its own input and output usage", () => {
  assert.equal(render({ prompt_eval_count: 8192, eval_count: 4096, num_ctx: 32768 }).width, "37.5%");
  assert.equal(render({ prompt_eval_count: 4096, eval_count: 2048, num_ctx: 32768 }).width, "18.75%");
});
