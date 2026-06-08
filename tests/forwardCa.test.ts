/**
 * Local MITM CA: persist a stable root, mint per-host leaves whose SAN matches
 * the requested host (so the agent's TLS validates), and reuse the root across
 * loads.
 */

import { mkdtempSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import forge from "node-forge";
import { describe, expect, it } from "vitest";

import { MitmCA, wildcardParent } from "../src/capture/forward/ca.js";

function sanNames(certPem: string): string[] {
  const cert = forge.pki.certificateFromPem(certPem.split("-----END CERTIFICATE-----")[0]! + "-----END CERTIFICATE-----\n");
  const ext = cert.getExtension("subjectAltName") as { altNames?: Array<{ value?: string }> } | null;
  return (ext?.altNames ?? []).map((a) => a.value ?? "").filter(Boolean);
}

describe("MitmCA", () => {
  it("computes a parent wildcard", () => {
    expect(wildcardParent("api2.cursor.sh")).toBe("*.cursor.sh");
    expect(wildcardParent("cursor.sh")).toBeNull();
  });

  it("persists a root CA and mints a leaf valid for the host", { timeout: 30000 }, () => {
    const dir = mkdtempSync(join(tmpdir(), "at-ca-"));
    const ca = MitmCA.loadOrCreate(dir);
    expect(existsSync(join(dir, "ca.pem"))).toBe(true);
    expect(existsSync(join(dir, "ca.key"))).toBe(true);

    const leaf = ca.leafFor("api2.cursor.sh");
    expect(leaf.key).toContain("PRIVATE KEY");
    expect(leaf.cert).toContain("BEGIN CERTIFICATE");
    const names = sanNames(leaf.cert);
    expect(names).toContain("api2.cursor.sh");
    expect(names).toContain("*.cursor.sh");

    // Cached: same host returns an identical leaf.
    expect(ca.leafFor("api2.cursor.sh").cert).toBe(leaf.cert);
  });

  it("reuses the persisted root across loads", { timeout: 30000 }, () => {
    const dir = mkdtempSync(join(tmpdir(), "at-ca-"));
    const first = MitmCA.loadOrCreate(dir).caCertPem;
    const onDisk = readFileSync(join(dir, "ca.pem"), "utf-8");
    const second = MitmCA.loadOrCreate(dir).caCertPem;
    expect(first).toBe(onDisk);
    expect(second).toBe(onDisk);
  });
});
