declare module 'react-plotly.js/factory' {
  import type { Component } from 'react'

  interface PlotParams {
    data: Array<Record<string, unknown>>
    layout?: Record<string, unknown>
    config?: Record<string, unknown>
    className?: string
    style?: React.CSSProperties
    [key: string]: unknown
  }

  function createPlotlyComponent(plotly: unknown): new (props: PlotParams) => Component<PlotParams>
  export default createPlotlyComponent
}

declare module 'plotly.js-dist-min' {
  const Plotly: unknown
  export default Plotly
}
