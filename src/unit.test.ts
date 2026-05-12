import { afterEach, expect, mock, test } from "bun:test";
import { executeNativeCompaction } from "./compact-client";
import {
	applyNativeCompactionRequestTemplate,
	createNativeCompactionRequestTemplate,
	isMatchingNativeCompactionRequestTemplate,
} from "./request-template";
import { buildCompactUrl } from "./runtime";

const baseModel = {
	provider: "openai",
	api: "openai-responses",
	id: "gpt-5-mini",
	name: "gpt-5-mini",
	baseUrl: "https://api.openai.com/v1",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100000,
	maxTokens: 1000,
};

let serializerImportCounter = 0;

async function loadSerializerModule() {
	mock.module("@mariozechner/pi-coding-agent", () => ({
		convertToLlm: (messages: unknown[]) => messages,
	}));
	return import(`./serializer.ts?unit=${serializerImportCounter++}`);
}

function createJwtWithAccountId(accountId: string): string {
	const header = Buffer.from(JSON.stringify({ alg: "none", typ: "JWT" })).toString("base64url");
	const payload = Buffer.from(
		JSON.stringify({
			"https://api.openai.com/auth": {
				chatgpt_account_id: accountId,
			},
		}),
	).toString("base64url");
	return `${header}.${payload}.signature`;
}

afterEach(() => {
	serializerImportCounter = 0;
	mock.restore();
});

test("buildCompactUrl uses codex compact path for openai-codex responses", () => {
	expect(buildCompactUrl("https://api.openai.com/v1", "openai-responses")).toBe(
		"https://api.openai.com/v1/responses/compact",
	);
	expect(buildCompactUrl("https://chatgpt.com/backend-api", "openai-codex-responses")).toBe(
		"https://chatgpt.com/backend-api/codex/responses/compact",
	);
	expect(buildCompactUrl("https://chatgpt.com/backend-api/codex", "openai-codex-responses")).toBe(
		"https://chatgpt.com/backend-api/codex/responses/compact",
	);
	expect(buildCompactUrl("https://chatgpt.com/backend-api/codex/responses", "openai-codex-responses")).toBe(
		"https://chatgpt.com/backend-api/codex/responses/compact",
	);
});

test("request template keeps only Codex v1 compact body fields", () => {
	const payload = {
		model: baseModel.id,
		instructions: "stale instructions must not be cached",
		input: [{ role: "user", content: [{ type: "input_text", text: "stale input must not be cached" }] }],
		tools: [{ type: "function", name: "lookup" }],
		parallel_tool_calls: false,
		reasoning: { effort: "high" },
		service_tier: "priority",
		prompt_cache_key: "cache-key",
		text: { verbosity: "low" },
		stream: true,
		store: true,
		include: ["reasoning.encrypted_content"],
		tool_choice: "auto",
		client_metadata: { source: "normal-response" },
		max_output_tokens: 1000,
		temperature: 0.4,
		previous_response_id: "resp_previous",
		prompt_cache_retention: "24h",
	};

	const template = createNativeCompactionRequestTemplate({
		identity: {
			provider: baseModel.provider,
			api: baseModel.api,
			model: baseModel.id,
			baseUrl: baseModel.baseUrl,
			sessionId: " session-template ",
		},
		payload,
	});

	expect(template).toEqual({
		identity: {
			provider: baseModel.provider,
			api: baseModel.api,
			model: baseModel.id,
			baseUrl: baseModel.baseUrl,
			sessionId: "session-template",
		},
		fields: {
			tools: payload.tools,
			parallel_tool_calls: false,
			reasoning: payload.reasoning,
			service_tier: "priority",
			prompt_cache_key: "cache-key",
			text: payload.text,
		},
	});
	expect(JSON.stringify(template?.fields)).not.toContain("stale instructions must not be cached");
	expect(JSON.stringify(template?.fields)).not.toContain("stale input must not be cached");
	expect(Object.hasOwn(template?.fields ?? {}, "stream")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "store")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "include")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "tool_choice")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "client_metadata")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "max_output_tokens")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "temperature")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "previous_response_id")).toBe(false);
	expect(Object.hasOwn(template?.fields ?? {}, "prompt_cache_retention")).toBe(false);
});

test("request template matching protects live compact model input and instructions", () => {
	const template = createNativeCompactionRequestTemplate({
		identity: {
			provider: baseModel.provider,
			api: baseModel.api,
			model: baseModel.id,
			baseUrl: baseModel.baseUrl,
			sessionId: "session-template",
		},
		payload: {
			model: baseModel.id,
			instructions: "cached instructions",
			input: [{ role: "user", content: "cached input" }],
			tools: [{ type: "function", name: "lookup" }],
			reasoning: { effort: "medium" },
		},
	});
	const liveRequest = {
		model: "live-model",
		instructions: "live instructions",
		input: [{ role: "user", content: [{ type: "input_text", text: "live input" }] }],
	};

	expect(
		isMatchingNativeCompactionRequestTemplate(template, {
			provider: baseModel.provider,
			api: baseModel.api,
			model: baseModel.id,
			baseUrl: baseModel.baseUrl,
			sessionId: "session-template",
		}),
	).toBe(true);
	expect(
		isMatchingNativeCompactionRequestTemplate(template, {
			provider: baseModel.provider,
			api: baseModel.api,
			model: "gpt-5-nano",
			baseUrl: baseModel.baseUrl,
			sessionId: "session-template",
		}),
	).toBe(false);
	expect(
		isMatchingNativeCompactionRequestTemplate(template, {
			provider: baseModel.provider,
			api: baseModel.api,
			model: baseModel.id,
			baseUrl: "https://api.openai.com/other",
			sessionId: "session-template",
		}),
	).toBe(false);
	expect(
		isMatchingNativeCompactionRequestTemplate(template, {
			provider: baseModel.provider,
			api: baseModel.api,
			model: baseModel.id,
			baseUrl: baseModel.baseUrl,
			sessionId: "different-session",
		}),
	).toBe(false);
	expect(applyNativeCompactionRequestTemplate(liveRequest as never, template)).toEqual({
		model: "live-model",
		instructions: "live instructions",
		input: [{ role: "user", content: [{ type: "input_text", text: "live input" }] }],
		tools: [{ type: "function", name: "lookup" }],
		reasoning: { effort: "medium" },
	});
});

test("executeNativeCompaction propagates resolved request headers and codex auth headers", async () => {
	const token = createJwtWithAccountId("acct_123");
	let fetchArgs: { url?: string; init?: RequestInit } = {};
	globalThis.fetch = mock(async (url: string | URL | Request, init?: RequestInit) => {
		fetchArgs = { url: String(url), init };
		return new Response(JSON.stringify({ output: [{ type: "compaction", encrypted_content: "opaque" }] }), {
			status: 200,
			headers: { "content-type": "application/json" },
		});
	}) as typeof fetch;

	const result = await executeNativeCompaction({
		sessionId: "session-compact-123",
		clientRequestId: "request-compact-456",
		runtime: {
			provider: "openai-codex",
			api: "openai-codex-responses",
			apiFamily: "openai-codex-responses",
			model: "gpt-5.1",
			baseUrl: "https://chatgpt.com/backend-api",
			apiKey: token,
			headers: {
				"x-test-model-header": "present",
				"x-test-runtime-header": "resolved",
			},
			compactPath: "codex/responses/compact",
			compactUrl: buildCompactUrl("https://chatgpt.com/backend-api", "openai-codex-responses"),
			currentModel: {
				...baseModel,
				provider: "openai-codex",
				api: "openai-codex-responses",
				id: "gpt-5.1",
				name: "gpt-5.1",
				baseUrl: "https://chatgpt.com/backend-api",
			},
		},
		request: {
			model: "gpt-5.1",
			instructions: "compact this",
			input: [{ role: "user", content: [{ type: "input_text", text: "hello" }] }],
		},
	});

	expect(result.ok).toBe(true);
	expect(fetchArgs.url).toBe("https://chatgpt.com/backend-api/codex/responses/compact");
	const headers = new Headers(fetchArgs.init?.headers);
	expect(headers.get("x-test-model-header")).toBe("present");
	expect(headers.get("x-test-runtime-header")).toBe("resolved");
	expect(headers.get("authorization")).toBe(`Bearer ${token}`);
	expect(headers.get("chatgpt-account-id")).toBe("acct_123");
	expect(headers.get("originator")).toBe("pi");
	expect(headers.get("openai-beta")).toBe("responses=experimental");
	expect(headers.get("session_id")).toBe("session-compact-123");
	expect(headers.get("x-client-request-id")).toBe("request-compact-456");
	expect(headers.get("content-type")).toBe("application/json");
});

test("serializer sanitizes unpaired surrogates in instructions and message content", async () => {
	const { serializeMessagesToCompactRequest, serializeMessagesToResponsesInput } = await loadSerializerModule();
	const invalid = "\ud800Hello\udc00";
	const request = serializeMessagesToCompactRequest({
		model: baseModel as never,
		instructions: `Prefix ${invalid}`,
		messages: [
			{ role: "user", content: [{ type: "text", text: invalid }], timestamp: 1 },
			{
				role: "assistant",
				provider: baseModel.provider,
				api: baseModel.api,
				model: baseModel.id,
				stopReason: "stop",
				content: [{ type: "text", text: invalid, textSignature: JSON.stringify({ v: 1, id: "msg_1" }) }],
				timestamp: 2,
			},
			{
				role: "toolResult",
				toolCallId: "call_1|fc_call_1",
				toolName: "read",
				isError: false,
				content: [{ type: "text", text: invalid }],
				timestamp: 3,
			},
		],
	});

	expect(JSON.stringify(request.instructions)).not.toContain("\\ud800");
	expect(JSON.stringify(request.input)).not.toContain("\\ud800");
	expect(JSON.stringify(request.input)).not.toContain("\\udc00");

	const inputOnly = serializeMessagesToResponsesInput(baseModel as never, [
		{ role: "user", content: [{ type: "text", text: invalid }], timestamp: 1 },
	] as never);
	expect(JSON.stringify(inputOnly)).not.toContain("\\ud800");
	expect(JSON.stringify(inputOnly)).not.toContain("\\udc00");
});
