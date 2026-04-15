export default function LoadingSpinner({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-gray-700 border-t-green-500 rounded-full animate-spin mx-auto" />
        <p className="mt-3 text-sm text-gray-500">{message}</p>
      </div>
    </div>
  )
}
