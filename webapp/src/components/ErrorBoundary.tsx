import { Component, type ReactNode, type ErrorInfo } from 'react'
import { Card, CardBody } from './ui/Card'
import { EmptyState } from './ui/EmptyState'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * Catches render-time errors anywhere below it in the tree and renders a
 * tokens-styled fallback Card with an EmptyState. The full stack is kept
 * available in a collapsed `<details>` for diagnosis without dominating
 * the screen on first load.
 *
 * Stays a class component because (a) React 19 still supports them and
 * (b) `componentDidCatch` / `getDerivedStateFromError` have no hooks
 * equivalent.
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info)
  }

  private handleReload = () => {
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      const message = this.state.error?.message ?? 'Unknown error'
      const stack = this.state.error?.stack ?? ''
      return (
        <div className="p-8 max-w-2xl mx-auto">
          <Card>
            <CardBody>
              <EmptyState
                icon="!"
                title="Something went wrong"
                description={message}
                action={
                  <button
                    type="button"
                    onClick={this.handleReload}
                    className="px-3 py-1.5 text-xs font-medium rounded bg-[var(--accent)]/20 hover:bg-[var(--accent)]/30 text-[var(--accent)] transition-colors"
                  >
                    Reload page
                  </button>
                }
              />
              {stack && (
                <details className="mt-4 text-xs text-[var(--text-tertiary)]">
                  <summary className="cursor-pointer select-none hover:text-[var(--text-secondary)]">
                    Show stack trace
                  </summary>
                  <pre className="mt-2 p-3 rounded bg-[var(--surface-3)] text-[var(--text-tertiary)] overflow-auto whitespace-pre-wrap font-mono">
                    {stack}
                  </pre>
                </details>
              )}
            </CardBody>
          </Card>
        </div>
      )
    }
    return this.props.children
  }
}
