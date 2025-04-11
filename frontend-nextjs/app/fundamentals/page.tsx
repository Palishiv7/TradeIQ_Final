'use client';

import Link from 'next/link';

export default function FundamentalsAssessment() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-6 bg-gray-50">
      <div className="w-full max-w-lg bg-white rounded-xl p-8 shadow-md text-center">
        <div className="w-16 h-16 rounded-full bg-green-100 text-green-600 text-3xl flex items-center justify-center mx-auto mb-4">
          📈
        </div>
        <h1 className="text-3xl font-bold mb-4">Market Fundamentals Assessment</h1>
        <p className="text-gray-600 mb-8">
          This assessment is currently under development. Check back soon!
        </p>
        <Link 
          href="/"
          className="py-2 px-6 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors inline-block"
        >
          Back to Assessments
        </Link>
      </div>
    </div>
  );
} 