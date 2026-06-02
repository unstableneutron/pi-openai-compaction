import type {
	BeforeProviderRequestEvent,
	ExtensionAPI,
	ExtensionContext,
	SessionBeforeCompactEvent,
} from "@mariozechner/pi-coding-agent";
import { executeNativeCompaction } from "./compact-client";
import { sanitizeCompactedWindow, summarizeCompactionOutputForDiagnostics } from "./compaction-output";
import { writeDebugArtifact } from "./debug";
import { findLatestNativeCompactionEntry, resolveLatestNativeCompactionEntry } from "./details-store";
import {
	normalizeNativeCompactedWindowForReplay,
	rewriteResponsesPayloadWithNativeReplay,
	serializeLiveTailToResponsesInput,
} from "./payload-rewrite";
import {
	applyNativeCompactionRequestTemplate,
	createNativeCompactionRequestTemplate,
	isMatchingNativeCompactionRequestTemplate,
	type NativeCompactionRequestTemplate,
	type NativeCompactionRequestTemplateIdentity,
} from "./request-template";
import { isResponsesCompatiblePayload, resolveNativeCompactionEnvironment, type ResponsesCompatibleRequestPayload } from "./runtime";
import { serializeMessagesToCompactRequest, type ResponsesInputItem } from "./serializer";
import { loadExtensionSettings } from "./settings";
import {
	createNativeCompactionDetails,
	createNativeCompactionShimResult,
	EXTENSION_ID,
	isNativeCompactionDetails,
	NATIVE_COMPACTION_DISPLAY_MESSAGE_TYPE,
	NATIVE_COMPACTION_DISPLAY_TEXT,
	type NativeCompactionRequestMeta,
} from "./types";

let latestRequestTemplate: NativeCompactionRequestTemplate | undefined;
let pendingPiCompactionNativeWindow: {
	window: ResponsesInputItem[];
	provider: string;
	api: string;
	baseUrl: string;
	sessionId?: string;
	sourceCompactionEntryId?: string;
} | undefined;

function isRecord(value: unknown): value is Record<string, unknown> {
	return !!value && typeof value === "object" && !Array.isArray(value);
}

function buildCompactionRequestMeta(event: SessionBeforeCompactEvent): NativeCompactionRequestMeta {
	return {
		tokensBefore: event.preparation.tokensBefore,
		previousSummaryPresent: Boolean(event.preparation.previousSummary),
	};
}

function getCurrentModelDebugInfo(ctx: ExtensionContext) {
	return ctx.model
		? {
			provider: ctx.model.provider,
			id: ctx.model.id,
		}
		: undefined;
}

function getCompactionIdentityDebugInfo(entry: { details?: unknown } | undefined) {
	return isNativeCompactionDetails(entry?.details)
		? {
			provider: entry.details.provider,
			api: entry.details.api,
			model: entry.details.model,
			baseUrl: entry.details.baseUrl,
		}
		: undefined;
}

function getSessionId(ctx: ExtensionContext): string | undefined {
	try {
		const sessionId = ctx.sessionManager.getSessionId();
		const normalized = sessionId?.trim();
		return normalized ? normalized : undefined;
	} catch {
		return undefined;
	}
}

function cloneCompactedWindow(window: readonly unknown[]): ResponsesInputItem[] | undefined {
	if (!window.every(isRecord)) return undefined;
	return window.map((item) => structuredClone(item) as ResponsesInputItem);
}

function stashLatestNativeWindowForPiCompactionFallback(args: {
	ctx: ExtensionContext;
	branchEntries: ReturnType<ExtensionContext["sessionManager"]["getBranch"]>;
	runtime: { provider: string; api: string; baseUrl: string };
}): boolean {
	pendingPiCompactionNativeWindow = undefined;
	const nativeEntry = resolveLatestNativeCompactionEntry(args.branchEntries, {
		provider: args.runtime.provider,
		api: args.runtime.api,
		baseUrl: args.runtime.baseUrl,
	});
	if (!nativeEntry.ok) return false;
	const compactedWindow = cloneCompactedWindow(normalizeNativeCompactedWindowForReplay(nativeEntry.entry.details.compactedWindow) ?? []);
	if (!compactedWindow || compactedWindow.length === 0) return false;
	pendingPiCompactionNativeWindow = {
		window: compactedWindow,
		provider: args.runtime.provider,
		api: args.runtime.api,
		baseUrl: args.runtime.baseUrl,
		sessionId: getSessionId(args.ctx),
		sourceCompactionEntryId: nativeEntry.entry.id,
	};
	return true;
}

function textFromResponsesContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((item) => isRecord(item) && item.type === "input_text" && typeof item.text === "string" ? item.text : "")
		.join("\n");
}

function isNativeCompactionDisplayMessage(message: unknown): boolean {
	return isRecord(message) && message.role === "custom" && message.customType === NATIVE_COMPACTION_DISPLAY_MESSAGE_TYPE;
}

function isPiCompactionSummarizationPayload(payload: ResponsesCompatibleRequestPayload): boolean {
	const instructions = typeof payload.instructions === "string" ? payload.instructions : "";
	if (/compact|summar/i.test(instructions)) return true;

	return payload.input.some((item) => {
		if (!isRecord(item)) return false;
		const role = item.role;
		const text = textFromResponsesContent(item.content);
		if ((role === "system" || role === "developer") && /compact|summar/i.test(text)) return true;
		if (role === "user" && /<conversation>|previous compaction summary|summary/i.test(text)) return true;
		return false;
	});
}

function injectPendingNativeWindowIntoPiCompactionRequest(args: {
	payload: unknown;
	ctx: ExtensionContext;
	runtime: { provider: string; api: string; baseUrl: string };
}): ResponsesCompatibleRequestPayload | undefined {
	const pending = pendingPiCompactionNativeWindow;
	if (!pending || pending.window.length === 0) return undefined;
	if (!isResponsesCompatiblePayload(args.payload)) return undefined;
	const sessionId = getSessionId(args.ctx);
	if (pending.sessionId && pending.sessionId !== sessionId) {
		pendingPiCompactionNativeWindow = undefined;
		return undefined;
	}
	if (!isPiCompactionSummarizationPayload(args.payload)) return undefined;
	if (pending.provider !== args.runtime.provider || pending.api !== args.runtime.api || pending.baseUrl !== args.runtime.baseUrl) {
		pendingPiCompactionNativeWindow = undefined;
		return undefined;
	}

	const input = [...args.payload.input];
	let insertAt = 0;
	while (insertAt < input.length) {
		const item = input[insertAt];
		if (!isRecord(item) || (item.role !== "system" && item.role !== "developer")) break;
		insertAt++;
	}

	pendingPiCompactionNativeWindow = undefined;
	return {
		...args.payload,
		input: [
			...input.slice(0, insertAt),
			...pending.window.map((item) => structuredClone(item)),
			...input.slice(insertAt),
		],
	};
}

function buildRequestTemplateIdentity(
	runtime: {
		provider: string;
		api: string;
		model: string;
		baseUrl: string;
	},
	ctx: ExtensionContext,
): NativeCompactionRequestTemplateIdentity {
	const sessionId = getSessionId(ctx);
	return {
		provider: runtime.provider,
		api: runtime.api,
		model: runtime.model,
		baseUrl: runtime.baseUrl,
		...(sessionId ? { sessionId } : {}),
	};
}

function buildCompactionInstructions(systemPrompt: string, customInstructions?: string): string {
	const guidance = customInstructions?.trim();
	if (!guidance) {
		return systemPrompt;
	}

	return `${systemPrompt}\n\nAdditional user guidance for this manual /compact request:\n${guidance}`;
}

async function handleSessionBeforeCompact(event: SessionBeforeCompactEvent, piContext: ExtensionContext) {
	const { settings } = loadExtensionSettings(piContext.cwd);
	if (!settings.enabled) {
		return undefined;
	}

	writeDebugArtifact(
		"compaction-event",
		{
			event: "session_before_compact",
			customInstructions: event.customInstructions,
			preparation: {
				tokensBefore: event.preparation.tokensBefore,
				firstKeptEntryId: event.preparation.firstKeptEntryId,
				previousSummaryPresent: Boolean(event.preparation.previousSummary),
				messagesToSummarizeCount: event.preparation.messagesToSummarize.length,
				turnPrefixMessagesCount: event.preparation.turnPrefixMessages.length,
			},
		},
		settings,
		piContext,
	);

	if (event.signal.aborted) {
		return { cancel: true };
	}

	const resolution = await resolveNativeCompactionEnvironment(piContext, {
		enabled: settings.enabled,
		supportedProviders: settings.supportedProviders,
		supportedApis: settings.supportedApis,
	});
	if (resolution.ok === false) {
		writeDebugArtifact(
			"compaction-event",
			{
				event: "session_before_compact.skip",
				reason: resolution.reason,
				provider: resolution.provider,
				api: resolution.api,
				model: resolution.model,
				baseUrl: resolution.baseUrl,
			},
			settings,
			piContext,
		);
		return undefined;
	}

	const runtime = resolution.runtime;
	const instructions = buildCompactionInstructions(piContext.getSystemPrompt(), event.customInstructions);
	const branchEntries = piContext.sessionManager.getBranch();
	const latestNativeCompaction = resolveLatestNativeCompactionEntry(branchEntries, {
		provider: runtime.provider,
		api: runtime.api,
		model: runtime.model,
		baseUrl: runtime.baseUrl,
	});

	let requestSource: "session-context" | "latest-native-replay" | "non-native-compaction-context";
	let request = undefined as ReturnType<typeof serializeMessagesToCompactRequest> | undefined;
	if (latestNativeCompaction.ok) {
		const compactedWindow = normalizeNativeCompactedWindowForReplay(latestNativeCompaction.entry.details.compactedWindow);
		if (!compactedWindow) {
			writeDebugArtifact(
				"compaction-event",
				{
					event: "session_before_compact.skip",
					reason: "invalid-compacted-window",
					provider: runtime.provider,
					api: runtime.api,
					model: runtime.model,
					baseUrl: runtime.baseUrl,
					latestCompactionIndex: latestNativeCompaction.index,
				},
				settings,
				piContext,
			);
			return undefined;
		}

		const liveTailEntries = branchEntries.slice(latestNativeCompaction.index + 1);
		requestSource = "latest-native-replay";
		request = {
			model: runtime.currentModel.id,
			input: [
				...compactedWindow,
				...serializeLiveTailToResponsesInput({ model: runtime.currentModel, entries: liveTailEntries }),
			],
			instructions,
		};
	} else {
		requestSource = latestNativeCompaction.reason === "no-compaction" ? "session-context" : "non-native-compaction-context";
		request = serializeMessagesToCompactRequest({
			model: runtime.currentModel,
			messages: piContext.sessionManager.buildSessionContext().messages,
			instructions,
		});
	}

	const requestTemplateIdentity = buildRequestTemplateIdentity(runtime, piContext);
	const requestTemplate = isMatchingNativeCompactionRequestTemplate(latestRequestTemplate, requestTemplateIdentity)
		? latestRequestTemplate
		: undefined;
	const compactRequest = applyNativeCompactionRequestTemplate(request, requestTemplate);

	const compactResult = await executeNativeCompaction({
		runtime,
		request: compactRequest,
		signal: event.signal,
		settings,
		context: piContext,
		sessionId: getSessionId(piContext),
	});

	if (compactResult.ok === false) {
		const stashed = compactResult.reason !== "aborted" && stashLatestNativeWindowForPiCompactionFallback({
			ctx: piContext,
			branchEntries,
			runtime,
		});
		writeDebugArtifact(
			"compaction-event",
			{
				event: "session_before_compact.native-failure",
				reason: compactResult.reason,
				status: compactResult.status,
				errorMessage: compactResult.errorMessage,
				pendingPiFallbackNativeWindow: stashed,
			},
			settings,
			piContext,
		);
		return compactResult.reason === "aborted" ? { cancel: true } : undefined;
	}

	const compactedWindow = sanitizeCompactedWindow(compactResult.compactedWindow);
	if (compactedWindow.length === 0) {
		writeDebugArtifact(
			"compaction-event",
			{
				event: "session_before_compact.invalid-native-output",
				reason: "no-installable-compacted-items",
				provider: runtime.provider,
				api: runtime.api,
				model: runtime.model,
				baseUrl: runtime.baseUrl,
				output: summarizeCompactionOutputForDiagnostics(compactResult.compactedWindow, compactedWindow),
			},
			settings,
			piContext,
		);
		return undefined;
	}

	let details: ReturnType<typeof createNativeCompactionDetails>;
	try {
		details = createNativeCompactionDetails({
			provider: runtime.provider,
			api: runtime.api,
			model: runtime.model,
			baseUrl: runtime.baseUrl,
			compactedWindow,
			compactResponseId: compactResult.compactResponseId,
			createdAt: compactResult.createdAt,
			requestMeta: buildCompactionRequestMeta(event),
		});
	} catch (error) {
		writeDebugArtifact(
			"compaction-event",
			{
				event: "session_before_compact.invalid-native-details",
				reason: error instanceof Error ? error.message : String(error),
				provider: runtime.provider,
				api: runtime.api,
				model: runtime.model,
				baseUrl: runtime.baseUrl,
			},
			settings,
			piContext,
		);
		return undefined;
	}
	const compaction = createNativeCompactionShimResult({
		firstKeptEntryId: event.preparation.firstKeptEntryId,
		tokensBefore: event.preparation.tokensBefore,
		details,
	});

	writeDebugArtifact(
		"compaction-event",
		{
			event: "session_before_compact.native-success",
			provider: runtime.provider,
			api: runtime.api,
			model: runtime.model,
			requestSource,
			requestTemplateFields: requestTemplate ? Object.keys(requestTemplate.fields).sort() : [],
			requestInputItems: compactRequest.input.length,
			compactResponseId: compactResult.compactResponseId,
			compactedItems: compactedWindow.length,
			firstKeptEntryId: event.preparation.firstKeptEntryId,
		},
		settings,
		piContext,
	);

	return { compaction };
}

async function handleBeforeProviderRequest(event: BeforeProviderRequestEvent, ctx: ExtensionContext) {
	const { settings } = loadExtensionSettings(ctx.cwd);
	if (!settings.enabled) {
		return undefined;
	}

	const resolution = await resolveNativeCompactionEnvironment(
		ctx,
		{
			enabled: settings.enabled,
			supportedProviders: settings.supportedProviders,
			supportedApis: settings.supportedApis,
		},
		event.payload,
	);
	if (resolution.ok === false) {
		writeDebugArtifact(
			"provider-request",
			{
				event: "before_provider_request.skip",
				reason: resolution.reason,
				provider: resolution.provider,
				api: resolution.api,
				model: resolution.model,
				baseUrl: resolution.baseUrl,
				currentModel: getCurrentModelDebugInfo(ctx),
				payload: event.payload,
			},
			settings,
			ctx,
		);
		return undefined;
	}

	const runtime = resolution.runtime;
	const piCompactionPayload = injectPendingNativeWindowIntoPiCompactionRequest({ payload: runtime.payload, ctx, runtime });
	if (piCompactionPayload !== undefined) return piCompactionPayload;

	const requestTemplateIdentity = buildRequestTemplateIdentity(runtime, ctx);
	const requestTemplate = requestTemplateIdentity.sessionId
		? createNativeCompactionRequestTemplate({
			identity: requestTemplateIdentity,
			payload: runtime.payload,
		})
		: undefined;
	if (requestTemplate) {
		latestRequestTemplate = requestTemplate;
	}

	const branchEntries = ctx.sessionManager.getBranch();
	const latestNativeCompaction = resolveLatestNativeCompactionEntry(branchEntries, {
		provider: runtime.provider,
		api: runtime.api,
		model: runtime.model,
		baseUrl: runtime.baseUrl,
	});
	const latestNativeCompactionEntry = latestNativeCompaction.ok
		? latestNativeCompaction.entry
		: latestNativeCompaction.reason === "latest-compaction-not-native"
			? findLatestNativeCompactionEntry(branchEntries, {
				provider: runtime.provider,
				api: runtime.api,
				model: runtime.model,
				baseUrl: runtime.baseUrl,
			})
			: undefined;
	if (!latestNativeCompactionEntry) {
		writeDebugArtifact(
			"provider-request",
			{
				event: "before_provider_request.no-native-compaction",
				reason: latestNativeCompaction.reason,
				provider: runtime.provider,
				api: runtime.api,
				model: runtime.model,
				baseUrl: runtime.baseUrl,
				branchEntries: branchEntries.length,
				latestCompactionIndex: latestNativeCompaction.latestCompactionIndex,
				latestCompactionIdentity: getCompactionIdentityDebugInfo(latestNativeCompaction.latestCompaction),
				payload: runtime.payload,
			},
			settings,
			ctx,
		);
		return undefined;
	}
	const rewrite = rewriteResponsesPayloadWithNativeReplay({
		model: runtime.currentModel,
		payload: runtime.payload,
		branchEntries,
		compactionEntry: latestNativeCompactionEntry,
	});
	if (!rewrite.ok) {
		writeDebugArtifact(
			"provider-request",
			{
				event: "before_provider_request.rewrite-failed",
				reason: rewrite.reason,
				provider: runtime.provider,
				api: runtime.api,
				model: runtime.model,
				baseUrl: runtime.baseUrl,
				compactionEntryId: latestNativeCompactionEntry.id,
				parity: rewrite.parity,
				payload: runtime.payload,
			},
			settings,
			ctx,
		);
		return undefined;
	}

	writeDebugArtifact(
		"provider-request",
		{
			event: "before_provider_request.native-rewrite",
			provider: runtime.provider,
			api: runtime.api,
			model: runtime.model,
			baseUrl: runtime.baseUrl,
			compactionEntryId: latestNativeCompactionEntry.id,
			boundaryIndex: rewrite.segments.boundaryIndex,
			firstKeptEntryIndex: rewrite.segments.firstKeptEntryIndex,
			originalInputItems: runtime.payload.input.length,
			rewrittenInputItems: rewrite.rewrittenPayload.input.length,
			freshPreambleItems: rewrite.segments.freshPreamble.length,
			trailingPreambleItems: rewrite.segments.trailingPreamble.length,
			compactionSummaryItems: rewrite.segments.compactionSummary.length,
			preCompactionKeptItems: rewrite.segments.preCompactionKeptWindow.input.length,
			compactedItems: rewrite.segments.compactedWindow.length,
			postCompactionTailItems: rewrite.segments.postCompactionTail.input.length,
			payload: rewrite.rewrittenPayload,
			originalPayload: runtime.payload,
		},
		settings,
		ctx,
	);

	return rewrite.rewrittenPayload;
}

export default function (pi: ExtensionAPI) {
	pi.on("session_start", (_event, ctx) => {
		const { settings, warnings } = loadExtensionSettings(ctx.cwd);
		if (!settings.enabled) return;

		if (warnings.length > 0 && ctx.hasUI && settings.debug) {
			ctx.ui.notify(`${EXTENSION_ID}: ${warnings[0]}`, "warning");
		}

		const artifactPath = writeDebugArtifact(
			"lifecycle",
			{
				event: "session_start",
				settings,
				warnings,
			},
			settings,
			ctx,
		);

		if (ctx.hasUI && (settings.notifyOnLoad || settings.debug)) {
			ctx.ui.notify(
				artifactPath
					? `${EXTENSION_ID} loaded • debug artifacts → ${artifactPath}`
					: `${EXTENSION_ID} loaded`,
				"info",
			);
		}
	});

	pi.on("session_before_compact", handleSessionBeforeCompact);
	pi.on("before_provider_request", handleBeforeProviderRequest);
	pi.on("session_compact", async (event, ctx) => {
		pendingPiCompactionNativeWindow = undefined;
		if (!event.fromExtension || !isNativeCompactionDetails(event.compactionEntry.details)) return;
		if (ctx.hasUI) {
			ctx.ui.notify(NATIVE_COMPACTION_DISPLAY_TEXT, "warning");
		}
	});
	pi.on("context", async (event) => ({ messages: event.messages.filter((message) => !isNativeCompactionDisplayMessage(message)) }));
}
