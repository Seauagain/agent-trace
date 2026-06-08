/**
 * Local certificate authority for the MITM forward proxy.
 *
 * Unlike `exec`/`capture` (which only redirect `*_BASE_URL` and never touch
 * TLS), the forward proxy intercepts HTTPS by terminating TLS with certificates
 * it mints on the fly. That only works if the agent trusts our root, so we
 * persist a stable root CA under `~/.agent-trace/ca` and expose its path for
 * `NODE_EXTRA_CA_CERTS` (Node clients) or the system trust store.
 *
 * Per-host leaf certs reuse a single key pair (RSA keygen is the slow part in
 * pure JS); only the cheap signing step runs per host, cached by SNI name.
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import forge from "node-forge";

const { pki, md } = forge;

export interface LeafCert {
  /** PEM private key (shared across leaves). */
  key: string;
  /** PEM certificate chain (leaf, valid for the requested host). */
  cert: string;
}

function defaultCaDir(): string {
  return join(homedir(), ".agent-trace", "ca");
}

function randomSerial(): string {
  // forge wants a positive hex serial; keep it short and leading-zero-safe.
  const bytes = forge.random.getBytesSync(16);
  return "00" + forge.util.bytesToHex(bytes);
}

function isIp(host: string): boolean {
  return /^\d{1,3}(\.\d{1,3}){3}$/.test(host) || host.includes(":");
}

/** Parent-wildcard for a host, so one leaf covers sibling subdomains too. */
export function wildcardParent(host: string): string | null {
  const parts = host.split(".");
  if (parts.length < 3) return null;
  return "*." + parts.slice(1).join(".");
}

export class MitmCA {
  private readonly caCert: forge.pki.Certificate;
  private readonly caKey: forge.pki.rsa.PrivateKey;
  private readonly leafKeys: forge.pki.rsa.KeyPair;
  private readonly leafKeyPem: string;
  private readonly leafCache = new Map<string, LeafCert>();
  /** Absolute path to the root CA certificate (PEM) for NODE_EXTRA_CA_CERTS. */
  readonly certPath: string;
  readonly caCertPem: string;

  private constructor(
    caCert: forge.pki.Certificate,
    caKey: forge.pki.rsa.PrivateKey,
    caCertPem: string,
    certPath: string,
  ) {
    this.caCert = caCert;
    this.caKey = caKey;
    this.caCertPem = caCertPem;
    this.certPath = certPath;
    // One shared leaf key pair: keygen is expensive, signing is cheap.
    this.leafKeys = pki.rsa.generateKeyPair(2048);
    this.leafKeyPem = pki.privateKeyToPem(this.leafKeys.privateKey);
  }

  /** Load a persisted root CA from `dir`, generating + saving one if absent. */
  static loadOrCreate(dir: string = defaultCaDir()): MitmCA {
    mkdirSync(dir, { recursive: true });
    const certPath = join(dir, "ca.pem");
    const keyPath = join(dir, "ca.key");

    if (existsSync(certPath) && existsSync(keyPath)) {
      const certPem = readFileSync(certPath, "utf-8");
      const keyPem = readFileSync(keyPath, "utf-8");
      return new MitmCA(
        pki.certificateFromPem(certPem),
        pki.privateKeyFromPem(keyPem) as forge.pki.rsa.PrivateKey,
        certPem,
        certPath,
      );
    }

    const keys = pki.rsa.generateKeyPair(2048);
    const cert = pki.createCertificate();
    cert.publicKey = keys.publicKey;
    cert.serialNumber = randomSerial();
    cert.validity.notBefore = new Date(Date.now() - 24 * 3600 * 1000);
    cert.validity.notAfter = new Date(Date.now() + 3650 * 24 * 3600 * 1000);
    const attrs = [{ name: "commonName", value: "agent-trace local CA" }, { name: "organizationName", value: "agent-trace" }];
    cert.setSubject(attrs);
    cert.setIssuer(attrs);
    cert.setExtensions([
      { name: "basicConstraints", cA: true, critical: true },
      { name: "keyUsage", keyCertSign: true, cRLSign: true, critical: true },
      { name: "subjectKeyIdentifier" },
    ]);
    cert.sign(keys.privateKey, md.sha256.create());

    const certPem = pki.certificateToPem(cert);
    const keyPem = pki.privateKeyToPem(keys.privateKey);
    writeFileSync(certPath, certPem, { mode: 0o644 });
    writeFileSync(keyPath, keyPem, { mode: 0o600 });
    return new MitmCA(cert, keys.privateKey, certPem, certPath);
  }

  /** Mint (and cache) a leaf certificate valid for `host`. */
  leafFor(host: string): LeafCert {
    const cached = this.leafCache.get(host);
    if (cached) return cached;

    const cert = pki.createCertificate();
    cert.publicKey = this.leafKeys.publicKey;
    cert.serialNumber = randomSerial();
    cert.validity.notBefore = new Date(Date.now() - 24 * 3600 * 1000);
    cert.validity.notAfter = new Date(Date.now() + 825 * 24 * 3600 * 1000);
    cert.setSubject([{ name: "commonName", value: host }]);
    cert.setIssuer(this.caCert.subject.attributes);

    const altNames: Array<{ type: number; value?: string; ip?: string }> = isIp(host)
      ? [{ type: 7, ip: host }]
      : [{ type: 2, value: host }];
    const wildcard = isIp(host) ? null : wildcardParent(host);
    if (wildcard) altNames.push({ type: 2, value: wildcard });

    cert.setExtensions([
      { name: "basicConstraints", cA: false },
      { name: "keyUsage", digitalSignature: true, keyEncipherment: true },
      { name: "extKeyUsage", serverAuth: true },
      { name: "subjectAltName", altNames },
    ]);
    cert.sign(this.caKey, md.sha256.create());

    const leaf: LeafCert = {
      key: this.leafKeyPem,
      cert: pki.certificateToPem(cert) + this.caCertPem,
    };
    this.leafCache.set(host, leaf);
    return leaf;
  }
}
