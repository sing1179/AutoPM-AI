export function Logo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <div className={`${className} relative flex items-center justify-center`}>
      <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        {/* PM Circle - Background */}
        <circle cx="12" cy="16" r="10" fill="black" stroke="white" strokeWidth="2"/>
        <text x="12" y="21" fontSize="10" fontWeight="bold" fill="white" textAnchor="middle">PM</text>
        
        {/* AI Circle - Foreground overlapping */}
        <circle cx="20" cy="16" r="10" fill="black" stroke="white" strokeWidth="2"/>
        <text x="20" y="21" fontSize="10" fontWeight="bold" fill="white" textAnchor="middle">AI</text>
      </svg>
    </div>
  );
}
