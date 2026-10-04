import assert from "node:assert/strict";

import {
  AuthenticationError,
  PermissionDeniedError,
  TypeSafeClient,
  TypeSafeError,
  UnprocessableEntityError,
  VERSION,
  choice,
  noul,
  score,
} from "@typesafe-ai/sdk";

const [baseURL] = process.argv.slice(2);
assert.ok(baseURL, "usage: node tests/typesafe_sdk_interop.mjs <base-url>");
assert.equal(VERSION, "0.6.0");

const serviceOrigin = new URL(baseURL).origin;
const nativeFetch = globalThis.fetch.bind(globalThis);
const requests = [];

const guardedFetch = async (input, init = {}) => {
  const url = new URL(typeof input === "string" ? input : input.url);
  assert.equal(url.origin, serviceOrigin, `unexpected SDK destination: ${url.origin}`);
  const headers = new Headers(init.headers);
  requests.push({
    method: init.method ?? "GET",
    path: url.pathname,
    authorization: headers.get("authorization"),
    sdk: headers.get("x-typesafe-sdk"),
  });
  return nativeFetch(input, init);
};

const fullAlphabetKey = "AZaz09-._~+/==";
const actualModel = "decider-4b-q4-k-m";
const clientFor = (apiKey) =>
  new TypeSafeClient({
    apiKey,
    baseURL,
    defaultModel: "jev-latest",
    fetch: guardedFetch,
    logLevel: "off",
    retry: { maxRetries: 0 },
    timeout: 5_000,
  });

const client = clientFor(fullAlphabetKey);
assert.equal(client.baseURL, baseURL);

const models = await client.models.list();
assert.deepEqual(
  new Set(models.map(({ name }) => name)),
  new Set([actualModel, "jev-latest"]),
);

const choiceResult = await client.systemOne({
  state: { incident: ["checkout", "unavailable"] },
  questions: {
    priority: choice(
      ["Choose", { timeframe: "today" }],
      { routine: null, urgent: { action: "now" } },
    ),
  },
});
assert.deepEqual(choiceResult.answers.priority, {
  type: "choice",
  choice: "urgent",
  confidence: 0.8,
  probabilities: { routine: 0.1, urgent: 0.9 },
});

const noulResult = await client.systemOne({
  state: ["Production", { status: "unavailable" }],
  questions: {
    isOutage: noul(
      { question: "Is production unavailable?" },
      { true: ["service", "down"], false: null },
    ),
  },
});
assert.deepEqual(noulResult.answers.isOutage, {
  type: "noul",
  noul: 0.9,
});

const scoreResult = await client.systemOne({
  state: "Checkout is unavailable.",
  questions: {
    severity: score(
      { question: ["Rate", "severity"] },
      ["low", { label: "high" }],
    ),
  },
});
assert.deepEqual(scoreResult.answers.severity, {
  type: "score",
  score: 0.9,
  confidence: 0.8,
  legend: { 0: "low", 1: { label: "high" } },
  probabilities: { 0: 0.1, 1: 0.9 },
});

const mixedResult = await client.systemOne({
  state: { message: "Production is unavailable." },
  questions: {
    priority: choice("Choose a priority.", { routine: null, urgent: "Act now" }),
    isOutage: noul("Is production unavailable?"),
    severity: score("Rate severity.", ["low", { label: "high" }]),
  },
});
assert.equal(mixedResult.model, actualModel);
assert.deepEqual(Object.keys(mixedResult.answers), [
  "priority",
  "isOutage",
  "severity",
]);
assert.equal(mixedResult.answers.priority.choice, "urgent");
assert.equal(mixedResult.answers.isOutage.noul, 0.9);
assert.equal(mixedResult.answers.severity.score, 0.9);
assert.deepEqual(mixedResult.usage, { input_tokens: 33, output_tokens: 3 });

const beforeOneLevel = requests.length;
await assert.rejects(
  async () =>
    client.systemOne({
      state: "evidence",
      questions: {
        classification: {
          type: "score",
          criteria: ["only level"],
        },
      },
    }),
  (error) => {
    assert.ok(error instanceof TypeSafeError);
    assert.match(error.message, /at least two scores are required/);
    return true;
  },
);
assert.equal(requests.length, beforeOneLevel, "one-level Score reached HTTP");

const oneLevelHTTP = await guardedFetch(`${baseURL}/v1/systemone`, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${fullAlphabetKey}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "jev-latest",
    state: "evidence",
    questions: {
      classification: { type: "score", criteria: ["only level"] },
    },
  }),
});
assert.equal(oneLevelHTTP.status, 200);
assert.deepEqual(await oneLevelHTTP.json(), {
  model: "decider-4b-q4-k-m",
  answers: {
    classification: {
      type: "score",
      score: 0,
      confidence: 1,
      legend: { 0: "only level" },
      probabilities: { 0: 1 },
    },
  },
  usage: { input_tokens: 0, output_tokens: 0 },
});

await assert.rejects(
  client.systemOne({
    state: null,
    questions: {
      priority: choice("Choose a priority.", { routine: null, urgent: null }),
    },
  }),
  (error) => {
    assert.ok(error instanceof UnprocessableEntityError);
    assert.equal(error.status, 422);
    assert.ok(error.body.detail.some(({ loc }) => loc.includes("state")));
    return true;
  },
);

await assert.rejects(clientFor("operator-probe-key").models.list(), (error) => {
  assert.ok(error instanceof AuthenticationError);
  assert.equal(error.status, 401);
  return true;
});

await assert.rejects(clientFor("rejected-caller-key").models.list(), (error) => {
  assert.ok(error instanceof PermissionDeniedError);
  assert.equal(error.status, 403);
  assert.deepEqual(error.body, { detail: "Backend rejected caller credentials." });
  return true;
});

const sdkRequests = requests.filter(
  ({ authorization, sdk }) =>
    authorization === `Bearer ${fullAlphabetKey}` && sdk === "typesafe-sdk/0.6.0",
);
assert.ok(sdkRequests.length >= 6);
assert.ok(requests.every(({ authorization }) => authorization?.startsWith("Bearer ")));

process.stdout.write(
  `${JSON.stringify({
    sdkVersion: VERSION,
    serviceOrigin,
    requestCount: requests.length,
    typedDecisions: ["choice", "noul", "score"],
    mixedDecision: true,
    oneLevelScoreRejectedLocally: true,
    oneLevelScoreAcceptedOverHTTP: true,
    nullStateRejectedByService: true,
    errors: [401, 403, 422],
  })}\n`,
);
