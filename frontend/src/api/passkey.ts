import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'
import { ADMIN_ME_KEY } from './admin'

export type PasskeyStatus = { enabled: boolean; has_credentials: boolean }

export type PasskeyCredential = {
  id: number
  label: string
  transports: string | null
  backup_eligible: boolean
  backup_state: boolean
  created_at: string
  last_used_at: string | null
}

const STATUS_KEY = ['passkey', 'status'] as const
const LIST_KEY = ['passkey', 'list'] as const

export function usePasskeyStatus() {
  return useQuery({
    queryKey: STATUS_KEY,
    queryFn: () => apiFetch<PasskeyStatus>('/sys/admin/passkey/status'),
    staleTime: 60_000,
    retry: false,
  })
}

export function usePasskeyList() {
  return useQuery({
    queryKey: LIST_KEY,
    queryFn: () => apiFetch<PasskeyCredential[]>('/sys/admin/passkey'),
    staleTime: 30_000,
    retry: false,
  })
}

export function useDeletePasskey() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) =>
      apiFetch<void>(`/sys/admin/passkey/${id}`, { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: LIST_KEY })
      qc.invalidateQueries({ queryKey: STATUS_KEY })
    },
  })
}

// ---------- WebAuthn binary helpers ----------
//
// PublicKeyCredential's binary fields (rawId, attestationObject, etc.) come
// in/out as ArrayBuffer. The browser API takes/returns BufferSource for
// challenge + allowCredentials.id; the server (py_webauthn) speaks
// base64url-encoded JSON. These helpers bridge the two.

function b64uToBytes(s: string): Uint8Array {
  const pad = '='.repeat((4 - (s.length % 4)) % 4)
  const b = atob(s.replace(/-/g, '+').replace(/_/g, '/') + pad)
  const out = new Uint8Array(b.length)
  for (let i = 0; i < b.length; i++) out[i] = b.charCodeAt(i)
  return out
}

function bytesToB64u(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf)
  let s = ''
  for (const b of bytes) s += String.fromCharCode(b)
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/** Decode a server-issued PublicKeyCredentialCreationOptionsJSON into the
 * BufferSource shape the browser API needs. */
function decodeCreationOptions(o: any): PublicKeyCredentialCreationOptions {
  return {
    ...o,
    challenge: b64uToBytes(o.challenge),
    user: { ...o.user, id: b64uToBytes(o.user.id) },
    excludeCredentials: (o.excludeCredentials ?? []).map((c: any) => ({
      ...c,
      id: b64uToBytes(c.id),
    })),
  }
}

function decodeRequestOptions(o: any): PublicKeyCredentialRequestOptions {
  return {
    ...o,
    challenge: b64uToBytes(o.challenge),
    allowCredentials: (o.allowCredentials ?? []).map((c: any) => ({
      ...c,
      id: b64uToBytes(c.id),
    })),
  }
}

function encodeRegistrationCredential(cred: PublicKeyCredential): any {
  const att = cred.response as AuthenticatorAttestationResponse
  return {
    id: cred.id,
    rawId: bytesToB64u(cred.rawId),
    type: cred.type,
    response: {
      attestationObject: bytesToB64u(att.attestationObject),
      clientDataJSON: bytesToB64u(att.clientDataJSON),
      transports: att.getTransports?.() ?? [],
    },
    authenticatorAttachment: cred.authenticatorAttachment ?? null,
    clientExtensionResults: cred.getClientExtensionResults?.() ?? {},
  }
}

function encodeAssertionCredential(cred: PublicKeyCredential): any {
  const a = cred.response as AuthenticatorAssertionResponse
  return {
    id: cred.id,
    rawId: bytesToB64u(cred.rawId),
    type: cred.type,
    response: {
      authenticatorData: bytesToB64u(a.authenticatorData),
      clientDataJSON: bytesToB64u(a.clientDataJSON),
      signature: bytesToB64u(a.signature),
      userHandle: a.userHandle ? bytesToB64u(a.userHandle) : null,
    },
    authenticatorAttachment: cred.authenticatorAttachment ?? null,
    clientExtensionResults: cred.getClientExtensionResults?.() ?? {},
  }
}

// ---------- Register flow ----------

export function useRegisterPasskey() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (label: string) => {
      const start = await apiFetch<{ challenge_id: string; publicKey: string }>(
        '/sys/admin/passkey/register/start',
        { method: 'POST' },
      )
      const options = decodeCreationOptions(JSON.parse(start.publicKey))
      const cred = (await navigator.credentials.create({
        publicKey: options,
      })) as PublicKeyCredential | null
      if (!cred) throw new Error('用户取消了 passkey 注册')
      return apiFetch<{ id: number; label: string; created_at: string }>(
        '/sys/admin/passkey/register/finish',
        {
          method: 'POST',
          body: JSON.stringify({
            challenge_id: start.challenge_id,
            label,
            credential: encodeRegistrationCredential(cred),
          }),
        },
      )
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: LIST_KEY })
      qc.invalidateQueries({ queryKey: STATUS_KEY })
    },
  })
}

// ---------- Login flow ----------

export function useLoginWithPasskey() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => {
      const start = await apiFetch<{ challenge_id: string; publicKey: string }>(
        '/sys/admin/passkey/login/start',
        { method: 'POST' },
      )
      const options = decodeRequestOptions(JSON.parse(start.publicKey))
      const cred = (await navigator.credentials.get({
        publicKey: options,
      })) as PublicKeyCredential | null
      if (!cred) throw new Error('用户取消了 passkey 登录')
      return apiFetch<{ ok: true; credential_label: string }>(
        '/sys/admin/passkey/login/finish',
        {
          method: 'POST',
          body: JSON.stringify({
            challenge_id: start.challenge_id,
            credential: encodeAssertionCredential(cred),
          }),
        },
      )
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ADMIN_ME_KEY })
    },
  })
}

export function isWebAuthnSupported(): boolean {
  return typeof window !== 'undefined' && !!window.PublicKeyCredential
}
