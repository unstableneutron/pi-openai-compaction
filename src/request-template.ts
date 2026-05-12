import type { ResponsesCompatibleRequestPayload } from "./runtime";
import type { NativeCompactionRequestBody } from "./serializer";
import type { NativeCompactionIdentity } from "./types";

export const NATIVE_COMPACTION_V1_TEMPLATE_FIELDS = [
	"tools",
	"parallel_tool_calls",
	"reasoning",
	"service_tier",
	"prompt_cache_key",
	"text",
] as const;

export type NativeCompactionV1TemplateField = (typeof NATIVE_COMPACTION_V1_TEMPLATE_FIELDS)[number];

export type NativeCompactionRequestTemplateFields = Partial<
	Pick<NativeCompactionRequestBody, NativeCompactionV1TemplateField>
>;

export type NativeCompactionRequestTemplateIdentity = NativeCompactionIdentity & {
	sessionId?: string;
};

export type NativeCompactionRequestTemplate = {
	identity: NativeCompactionRequestTemplateIdentity;
	fields: NativeCompactionRequestTemplateFields;
};

function normalizeOptionalString(value: string | undefined): string | undefined {
	const normalized = value?.trim();
	return normalized ? normalized : undefined;
}

function cloneTemplateValue(value: unknown): { ok: true; value: unknown } | { ok: false } {
	try {
		return { ok: true, value: structuredClone(value) };
	} catch {
		return { ok: false };
	}
}

function cloneTemplateFields(fields: NativeCompactionRequestTemplateFields): NativeCompactionRequestTemplateFields {
	const cloned: NativeCompactionRequestTemplateFields = {};

	for (const field of NATIVE_COMPACTION_V1_TEMPLATE_FIELDS) {
		if (!Object.hasOwn(fields, field)) {
			continue;
		}

		const clone = cloneTemplateValue(fields[field]);
		if (clone.ok) {
			cloned[field] = clone.value;
		}
	}

	return cloned;
}

export function createNativeCompactionRequestTemplate(args: {
	identity: NativeCompactionRequestTemplateIdentity;
	payload: ResponsesCompatibleRequestPayload;
}): NativeCompactionRequestTemplate | undefined {
	const fields: NativeCompactionRequestTemplateFields = {};

	for (const field of NATIVE_COMPACTION_V1_TEMPLATE_FIELDS) {
		if (!Object.hasOwn(args.payload, field) || args.payload[field] === undefined) {
			continue;
		}

		const clone = cloneTemplateValue(args.payload[field]);
		if (clone.ok) {
			fields[field] = clone.value;
		}
	}

	if (Object.keys(fields).length === 0) {
		return undefined;
	}

	return {
		identity: {
			provider: args.identity.provider,
			api: args.identity.api,
			model: args.identity.model,
			baseUrl: args.identity.baseUrl,
			...(normalizeOptionalString(args.identity.sessionId)
				? { sessionId: normalizeOptionalString(args.identity.sessionId) }
				: {}),
		},
		fields,
	};
}

export function isMatchingNativeCompactionRequestTemplate(
	template: NativeCompactionRequestTemplate | undefined,
	identity: NativeCompactionRequestTemplateIdentity,
): template is NativeCompactionRequestTemplate {
	return (
		!!template &&
		template.identity.provider === identity.provider &&
		template.identity.api === identity.api &&
		template.identity.model === identity.model &&
		template.identity.baseUrl === identity.baseUrl &&
		(template.identity.sessionId ?? undefined) === (normalizeOptionalString(identity.sessionId) ?? undefined)
	);
}

export function applyNativeCompactionRequestTemplate(
	request: NativeCompactionRequestBody,
	template: NativeCompactionRequestTemplate | undefined,
): NativeCompactionRequestBody {
	if (!template) {
		return request;
	}

	return {
		...request,
		...cloneTemplateFields(template.fields),
	};
}
