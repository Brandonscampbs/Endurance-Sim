import { Card, CardBody, CardHeader, CardTitle } from '../../components/ui'
import { useSimulateStore } from '../../stores/simulateStore'

export default function RunNotes() {
  const result = useSimulateStore((s) => s.result)
  const notes = result?.notes ?? []
  if (notes.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle>Notes</CardTitle>
      </CardHeader>
      <CardBody>
        <ul className="list-disc pl-5 space-y-1.5 text-sm text-[var(--text-secondary)]">
          {notes.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
        </ul>
      </CardBody>
    </Card>
  )
}
