/** Decoder registry for the MITM forward proxy. */

import { cursorDecoder } from "./cursor.js";
import { jsonDecoder } from "./json.js";
import type { DecodedCapture, HttpExchange, WireDecoder } from "./types.js";

export type { DecodedCapture, HttpExchange, WireDecoder } from "./types.js";
export { cursorDecoder } from "./cursor.js";
export { jsonDecoder } from "./json.js";

/** Default decoders, tried in order. Cursor (protobuf) before generic JSON. */
export function defaultDecoders(): WireDecoder[] {
  return [cursorDecoder, jsonDecoder];
}

/** Run the first decoder that recognizes the exchange. */
export function decodeExchange(ex: HttpExchange, decoders: WireDecoder[]): DecodedCapture | null {
  for (const d of decoders) {
    try {
      if (d.matches(ex)) {
        const decoded = d.decode(ex);
        if (decoded) return decoded;
      }
    } catch {
      /* a decoder throwing must not take down the proxy */
    }
  }
  return null;
}
