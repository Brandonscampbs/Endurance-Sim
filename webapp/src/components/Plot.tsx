import factory from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'

// react-plotly.js/factory is CJS — Vite may deliver { default: fn } as the default import
const createPlotlyComponent = typeof factory === 'function' ? factory : (factory as any).default
const Plot = createPlotlyComponent(Plotly)
export default Plot
