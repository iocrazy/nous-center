import { useNavigate, useParams } from 'react-router-dom'
import ServiceDetail from './ServiceDetail'

/** Wrapper that lifts the :id param out of the URL and passes the
 *  back-button handler. Kept tiny so testing ServiceDetail doesn't drag
 *  in BrowserRouter. */
export default function ServiceDetailRoute() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  if (!id) return null
  return <ServiceDetail serviceId={id} onBack={() => navigate('/services')} />
}
