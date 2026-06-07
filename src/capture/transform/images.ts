/**
 * Image content conversion helpers. Google / Responses converters are omitted
 * since those API families are not yet supported by the proxy.
 */

type Dict = Record<string, unknown>;

const DATA_URL_RE = /^data:([^;,]+);base64,(.*)$/s;
// OpenAI's image_url.detail only accepts these; drop unknowns (else vLLM 400s).
const VALID_IMAGE_DETAILS = new Set(["auto", "low", "high"]);

function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function isImageMimeType(mimeType: unknown): boolean {
  return typeof mimeType === "string" && mimeType.toLowerCase().startsWith("image/");
}

export function makeDataUrl(mimeType: string, data: string): string {
  if (data.startsWith("data:")) return data;
  return `data:${mimeType};base64,${data}`;
}

export function parseDataUrl(url: string): [string, string] | null {
  const match = DATA_URL_RE.exec(url);
  if (!match) return null;
  const mimeType = match[1]!;
  if (!isImageMimeType(mimeType)) return null;
  return [mimeType, match[2]!];
}

export function openaiImageUrlBlock(url: string, detail?: unknown): Dict {
  const imageUrl: Dict = { url };
  if (typeof detail === "string" && VALID_IMAGE_DETAILS.has(detail)) imageUrl["detail"] = detail;
  return { type: "image_url", image_url: imageUrl };
}

export function openaiTextBlock(text: string): Dict {
  return { type: "text", text };
}

export function openaiImageUrl(block: Dict): string | null {
  const imageUrl = block["image_url"];
  if (typeof imageUrl === "string" && imageUrl) return imageUrl;
  if (isDict(imageUrl)) {
    const url = imageUrl["url"];
    if (typeof url === "string" && url) return url;
  }
  return null;
}

export function openaiContentFromTextAndImages(parts: Dict[], textSeparator = "\n"): string | Dict[] {
  const hasImage = parts.some((part) => part["type"] === "image_url");
  if (hasImage) return parts;
  return parts
    .filter((part) => part["type"] === "text" && typeof part["text"] === "string")
    .map((part) => part["text"] as string)
    .join(textSeparator);
}

export function anthropicContentToOpenaiChat(content: unknown): string | Dict[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content ? String(content) : "";

  const parts: Dict[] = [];
  for (const block of content) {
    if (typeof block === "string") {
      parts.push(openaiTextBlock(block));
      continue;
    }
    if (!isDict(block)) continue;

    const blockType = block["type"];
    if (blockType === "text") {
      const text = block["text"];
      if (typeof text === "string") parts.push(openaiTextBlock(text));
    } else if (blockType === "image") {
      const image = anthropicImageToOpenaiChat(block);
      if (image) parts.push(image);
    } else if (blockType === "document") {
      const text = anthropicDocumentToText(block);
      if (text) parts.push(openaiTextBlock(text));
    }
  }
  return openaiContentFromTextAndImages(parts);
}

export function anthropicDocumentToText(block: Dict): string {
  const source = block["source"];
  if (!isDict(source)) return "";
  const sourceType = source["type"];
  if (sourceType === "text") {
    const data = source["data"];
    return typeof data === "string" ? data : "";
  }
  if (sourceType === "content") {
    const inner = source["content"];
    if (Array.isArray(inner)) {
      const pieces: string[] = [];
      for (const innerBlock of inner) {
        if (isDict(innerBlock) && innerBlock["type"] === "text") {
          const text = innerBlock["text"];
          if (typeof text === "string") pieces.push(text);
        }
      }
      return pieces.join("\n");
    }
  }
  return "";
}

export function anthropicImageToOpenaiChat(block: Dict): Dict | null {
  const source = block["source"];
  if (!isDict(source)) return null;

  const sourceType = source["type"];
  if (sourceType === "base64") {
    const mimeType = source["media_type"] ?? source["mediaType"];
    const data = source["data"];
    if (isImageMimeType(mimeType) && typeof data === "string" && data) {
      return openaiImageUrlBlock(makeDataUrl(mimeType as string, data));
    }
  }
  if (sourceType === "url") {
    const url = source["url"];
    if (typeof url === "string" && url) return openaiImageUrlBlock(url);
  }
  return null;
}

export function openaiChatContentToAnthropicBlocks(content: unknown): Dict[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content)) return [{ type: "text", text: content ? String(content) : "" }];

  const blocks: Dict[] = [];
  for (const part of content) {
    if (typeof part === "string") {
      blocks.push({ type: "text", text: part });
      continue;
    }
    if (!isDict(part)) continue;
    const partType = part["type"];
    if (partType === "text") {
      const text = part["text"];
      if (typeof text === "string") blocks.push({ type: "text", text });
    } else if (partType === "image_url") {
      const image = openaiChatImageToAnthropic(part);
      if (image) blocks.push(image);
    }
  }
  return blocks.length > 0 ? blocks : [{ type: "text", text: "" }];
}

export function openaiChatImageToAnthropic(part: Dict): Dict | null {
  const url = openaiImageUrl(part);
  if (!url) return null;
  const parsed = parseDataUrl(url);
  if (parsed) {
    const [mimeType, data] = parsed;
    return { type: "image", source: { type: "base64", media_type: mimeType, data } };
  }
  return { type: "image", source: { type: "url", url } };
}
