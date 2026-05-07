import { Card, CardBody, EmptyState } from '../../components/ui'

export default function SimulateEmpty() {
  return (
    <Card>
      <CardBody>
        <EmptyState
          title="No simulation has been run yet."
          description="Adjust the tune parameters above and click Run to compute a full Michigan 2025 endurance with your overrides. A run takes about 10 seconds."
        />
      </CardBody>
    </Card>
  )
}
