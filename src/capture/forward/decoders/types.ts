/**
 * Wire decoders for the MITM forward proxy.
 *
 * The forward proxy hands each fully-buffered HTTPS exchange to a registry of
 * decoders. A decoder that recognizes the traffic returns a normalized
 * OpenAI-chat (request, response) pair, which is stored exactly like a
 * passthrough capture so the existing `build` pipeline produces SFT/RL samples.
 * Traffic no decoder claims (npm, telemetry, ...) is ignored.
 */

export type Dict = Record<string, unknown>;

export interface HttpExchange {
  /** SNI host the client connected to (e.g. `api2.cursor.sh`). */
  host: string;
  method: string;
  /** Request path incl. query (`:path` for h2). */
  path: string;
  reqHeaders: Record<string, string>;
  reqBody: Buffer;
  status: number;
  resHeaders: Record<string, string>;
  resBody: Buffer;
}

export interface DecodedCapture {
  /** OpenAI-chat-shaped request (`messages`, `model`, ...). */
  request: Dict;
  /** OpenAI-chat-shaped response (`choices[0].message`). */
  response: Dict;
  model: string;
  /** Logical API family for metadata (`cursor_agent`, `openai_chat`, ...). */
  apiType: string;
  metadata?: Dict;
}

export interface WireDecoder {
  readonly name: string;
  matches(ex: HttpExchange): boolean;
  decode(ex: HttpExchange): DecodedCapture | null;
}

export function headerValue(
  headers: Record<string, string>,
  name: string,
): string | undefined {
  const lower = name.toLowerCase();
  for (const [k, v] of Object.entries(headers)) {
    if (k.toLowerCase() === lower) return v;
  }
  return undefined;
}
