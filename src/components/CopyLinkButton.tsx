import { useState } from 'react'

/** Copies the current URL, which encodes all inputs, so the result can be shared. */
export function CopyLinkButton() {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard can be blocked (e.g. insecure context); the address bar still has the link.
      window.prompt('Copy this link:', window.location.href)
    }
  }

  return (
    <button
      type="button"
      onClick={copy}
      className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium shadow-sm hover:bg-slate-50 focus:ring-2 focus:ring-indigo-200 focus:outline-none"
    >
      {copied ? 'Link copied' : 'Copy link'}
    </button>
  )
}
