import { Link } from 'react-router-dom'
import { Compass } from 'lucide-react'
import PageTransition from '../components/ui/PageTransition'
import { Card } from '../components/ui/Card'
import Button from '../components/ui/Button'

export default function NotFound() {
  return (
    <PageTransition title="Not found">
      <Card index={0} hover={false}>
        <div className="cv-empty">
          <Compass size={30} strokeWidth={1.5} />
          <strong>That route does not exist</strong>
          <p>Pick one of the six workspaces from the sidebar.</p>
          <Link to="/overview">
            <Button variant="primary">Go to Overview</Button>
          </Link>
        </div>
      </Card>
    </PageTransition>
  )
}
