/** Detect incoming API type from request path, headers, and body. */

export enum APIType {
  ANTHROPIC = "anthropic",
  OPENAI_CHAT = "openai_chat",
  OPENAI_RESPONSES = "openai_responses",
  GOOGLE = "google",
}

type Dict = Record<string, unknown>;

/** Detect API type. Priority: path -> headers -> body structure. */
export function detect(path: string, headers: Record<string, string>, body: Dict): APIType {
  // Path-based detection (most reliable).
  if (path.includes("/v1/messages")) return APIType.ANTHROPIC;
  if (path.includes("/v1/chat/completions")) return APIType.OPENAI_CHAT;
  if (path.includes("/v1/responses")) return APIType.OPENAI_RESPONSES;
  if (path.includes("generateContent")) return APIType.GOOGLE;

  // Header-based detection.
  const lower: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) lower[k.toLowerCase()] = v;
  if ("anthropic-version" in lower) return APIType.ANTHROPIC;

  // Body-based detection (fallback).
  if ("contents" in body) return APIType.GOOGLE;
  if ("input" in body && "instructions" in body) return APIType.OPENAI_RESPONSES;

  return APIType.OPENAI_CHAT;
}

/** Extract the model name from the request body based on API type. */
export function extractModel(apiType: APIType, body: Dict): string {
  if (apiType === APIType.GOOGLE) {
    return (body["model"] as string) ?? "gemini-pro";
  }
  return (body["model"] as string) ?? "unknown";
}
