"use client";

import { useEffect, useRef, useState } from "react";

type CopyButtonProps = {
  analyticsEvent?: string;
  label?: string;
  text: string;
};

export function CopyButton({
  label = "Copy",
  text,
}: CopyButtonProps) {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (resetTimer.current) clearTimeout(resetTimer.current);
    },
    [],
  );

  async function copy() {
    try {
      if (!navigator.clipboard) throw new Error("Clipboard API unavailable");
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
    if (resetTimer.current) clearTimeout(resetTimer.current);
    resetTimer.current = setTimeout(() => setStatus("idle"), 2200);
  }

  return (
    <button
      className="copy-button"
      type="button"
      onClick={copy}
      aria-label={`${label}: ${text}`}
    >
      <span aria-live="polite">
        {status === "copied" ? "Copied" : status === "failed" ? "Select text" : label}
      </span>
    </button>
  );
}
