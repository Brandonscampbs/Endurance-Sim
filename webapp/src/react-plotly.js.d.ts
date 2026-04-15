declare module 'react-plotly.js' {
  import type { Component } from 'react'

  interface PlotParams {
    data: Plotly.Data[]
    layout?: Partial<Plotly.Layout>
    config?: Partial<Plotly.Config>
    frames?: Plotly.Frame[]
    revision?: number
    onInitialized?: (figure: Readonly<{ data: Plotly.Data[]; layout: Partial<Plotly.Layout> }>, graphDiv: HTMLElement) => void
    onUpdate?: (figure: Readonly<{ data: Plotly.Data[]; layout: Partial<Plotly.Layout> }>, graphDiv: HTMLElement) => void
    onPurge?: (figure: Readonly<{ data: Plotly.Data[]; layout: Partial<Plotly.Layout> }>, graphDiv: HTMLElement) => void
    onError?: (err: Error) => void
    onRelayout?: (event: Plotly.PlotRelayoutEvent) => void
    onClick?: (event: Plotly.PlotMouseEvent) => void
    className?: string
    style?: React.CSSProperties
    useResizeHandler?: boolean
    debug?: boolean
    divId?: string
  }

  class Plot extends Component<PlotParams> {}
  export default Plot
}

declare namespace Plotly {
  interface Data {
    type?: string
    mode?: string
    x?: number[] | string[]
    y?: number[] | string[]
    z?: number[] | string[]
    name?: string
    marker?: Partial<PlotMarker>
    line?: Partial<ScatterLine>
    hovertemplate?: string
    text?: string | string[]
    [key: string]: unknown
  }

  interface PlotMarker {
    color: string | number | number[] | string[]
    colorscale: ColorScale
    cmin: number
    cmax: number
    size: number | number[]
    symbol: string
    opacity: number
    colorbar: Partial<ColorBar>
    [key: string]: unknown
  }

  interface ColorBar {
    title: string
    tickfont: Partial<Font>
    titlefont: Partial<Font>
    [key: string]: unknown
  }

  type ColorScale = string | [number, string][]

  interface ScatterLine {
    color: string
    width: number
    dash: string
    [key: string]: unknown
  }

  interface Font {
    family: string
    size: number
    color: string
  }

  interface Layout {
    title: string | Partial<{ text: string; font: Partial<Font>; x: number; y: number }>
    paper_bgcolor: string
    plot_bgcolor: string
    xaxis: Partial<LayoutAxis>
    yaxis: Partial<LayoutAxis>
    margin: Partial<Margin>
    showlegend: boolean
    legend: Partial<Legend>
    hovermode: string | false
    autosize: boolean
    width: number
    height: number
    [key: string]: unknown
  }

  interface LayoutAxis {
    title: string | Partial<{ text: string; font: Partial<Font> }>
    visible: boolean
    color: string
    gridcolor: string
    zerolinecolor: string
    scaleanchor: string
    scaleratio: number
    range: [number, number]
    [key: string]: unknown
  }

  interface Margin {
    t: number
    b: number
    l: number
    r: number
    pad: number
  }

  interface Legend {
    font: Partial<Font>
    bgcolor: string
    x: number
    y: number
    xanchor: string
    yanchor: string
    [key: string]: unknown
  }

  interface Config {
    responsive: boolean
    displayModeBar: boolean
    displaylogo: boolean
    scrollZoom: boolean
    [key: string]: unknown
  }

  interface Frame {
    name: string
    data: Data[]
    group: string
  }

  interface PlotRelayoutEvent {
    [key: string]: unknown
  }

  interface PlotMouseEvent {
    points: PlotDatum[]
    event: MouseEvent
  }

  interface PlotDatum {
    x: number | string
    y: number | string
    pointIndex: number
    data: Data
    [key: string]: unknown
  }
}
