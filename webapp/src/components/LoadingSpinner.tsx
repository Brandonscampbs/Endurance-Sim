/**
 * Lightweight loading spinner used by pages that don't need a full
 * skeleton placeholder. Tokens-driven so it matches the rest of the
 * design system. Public API (`message?: string`) is unchanged.
 */
export default function LoadingSpinner({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="text-center">
        <div
          className="w-8 h-8 border-2 border-[var(--border-subtle)] border-t-[var(--accent)] rounded-full animate-spin mx-auto"
          role="status"
          aria-label="Loading"
        />
        <p className="mt-3 text-sm text-[var(--text-tertiary)] animate-pulse">
          {message}
        </p>
      </div>
    </div>
  )
}
