'use client';

import React, { useState, useEffect } from 'react';
import CandlestickQuiz, { CandlestickQuizQuestion } from '../../components/CandlestickQuiz';
import Leaderboard, { LeaderboardEntry } from '../../components/Leaderboard';

// Mock authentication (in a real app, this would be handled by proper auth)
const MOCK_AUTH_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.XbPfbIHMI6arZ3Y922BhjWgQzWXcXNrz0ogtVhfEd2o';
const MOCK_USER_ID = 'user123';

// Mock assessment ID
const MOCK_ASSESSMENT_ID = 'candlestick-patterns-101';

// Mock quiz questions with sample data
const MOCK_QUESTIONS: CandlestickQuizQuestion[] = [
  {
    id: 'q1',
    chartData: [
      { time: '2023-01-01', open: 100, high: 105, low: 98, close: 103, volume: 10000 },
      { time: '2023-01-02', open: 103, high: 108, low: 102, close: 107, volume: 12000 },
      { time: '2023-01-03', open: 107, high: 110, low: 106, close: 109, volume: 15000 },
      { time: '2023-01-04', open: 109, high: 112, low: 108, close: 110, volume: 11000 },
      { time: '2023-01-05', open: 110, high: 115, low: 109, close: 114, volume: 18000 },
      { time: '2023-01-06', open: 114, high: 119, low: 113, close: 116, volume: 16000 },
      { time: '2023-01-07', open: 116, high: 116, low: 110, close: 110, volume: 20000 },
      { time: '2023-01-08', open: 110, high: 112, low: 105, close: 105, volume: 18000 },
      { time: '2023-01-09', open: 105, high: 105, low: 95, close: 96, volume: 25000 },
      { time: '2023-01-10', open: 96, high: 98, low: 94, close: 95, volume: 15000 },
    ],
    patterns: [
      {
        id: 'p1',
        name: 'Bullish Trend',
        description: 'A series of higher highs and higher lows',
        reliability: 'high',
        bullish: true,
        bearish: false,
      },
      {
        id: 'p2',
        name: 'Bearish Reversal',
        description: 'A change from an uptrend to a downtrend',
        reliability: 'medium',
        bullish: false,
        bearish: true,
      },
      {
        id: 'p3',
        name: 'Evening Star',
        description: 'A bearish reversal pattern consisting of three candles',
        reliability: 'high',
        bullish: false,
        bearish: true,
      },
      {
        id: 'p4',
        name: 'Hammer',
        description: 'A single candle pattern with a small body and long lower shadow',
        reliability: 'medium',
        bullish: true,
        bearish: false,
      },
    ],
    correctPatternIds: ['p2', 'p3'],
    explanation: 'This chart shows a clear bearish reversal with an evening star pattern forming on days 6-8, indicating a potential downtrend.',
    difficulty: 'beginner',
    timeLimit: 60,
  },
  {
    id: 'q2',
    chartData: [
      { time: '2023-02-01', open: 80, high: 82, low: 79, close: 81, volume: 12000 },
      { time: '2023-02-02', open: 81, high: 83, low: 80, close: 82, volume: 13000 },
      { time: '2023-02-03', open: 82, high: 84, low: 81, close: 83, volume: 15000 },
      { time: '2023-02-04', open: 83, high: 85, low: 83, close: 84, volume: 16000 },
      { time: '2023-02-05', open: 84, high: 86, low: 83.5, close: 85, volume: 17000 },
      { time: '2023-02-06', open: 85, high: 86, low: 83, close: 83.5, volume: 20000 },
      { time: '2023-02-07', open: 83.5, high: 84, low: 82, close: 82.5, volume: 18000 },
      { time: '2023-02-08', open: 82.5, high: 83, low: 81, close: 81.5, volume: 19000 },
      { time: '2023-02-09', open: 81.5, high: 84, low: 81, close: 83.5, volume: 22000 },
      { time: '2023-02-10', open: 83.5, high: 86, low: 83, close: 85.5, volume: 24000 },
    ],
    patterns: [
      {
        id: 'p5',
        name: 'Double Bottom',
        description: 'A bullish reversal pattern with two lows at approximately the same level',
        reliability: 'high',
        bullish: true,
        bearish: false,
      },
      {
        id: 'p6',
        name: 'Head and Shoulders',
        description: 'A bearish reversal pattern consisting of three peaks',
        reliability: 'high',
        bullish: false,
        bearish: true,
      },
      {
        id: 'p7',
        name: 'Bullish Engulfing',
        description: 'A bullish reversal pattern where a small bearish candle is engulfed by a larger bullish candle',
        reliability: 'medium',
        bullish: true,
        bearish: false,
      },
      {
        id: 'p8',
        name: 'Doji',
        description: 'A candle with a small body, indicating indecision',
        reliability: 'low',
        bullish: false,
        bearish: false,
      },
    ],
    correctPatternIds: ['p5', 'p7'],
    explanation: 'The chart shows a clear double bottom pattern (days 7 and 9) followed by a bullish engulfing pattern (days 9-10), indicating a potential uptrend.',
    difficulty: 'intermediate',
    timeLimit: 90,
  },
];

// Mock leaderboard entries
const MOCK_LEADERBOARD: LeaderboardEntry[] = [
  {
    userId: 'user456',
    name: 'Jane Smith',
    rank: 1,
    score: 950,
    avatarUrl: 'https://randomuser.me/api/portraits/women/32.jpg',
    badges: ['🔥', '🏆'],
    lastActive: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
  },
  {
    userId: 'user789',
    name: 'Robert Johnson',
    rank: 2,
    score: 920,
    avatarUrl: 'https://randomuser.me/api/portraits/men/47.jpg',
    badges: ['🏆'],
    lastActive: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
  },
  {
    userId: 'user123',
    name: 'John Doe',
    rank: 3,
    score: 880,
    avatarUrl: 'https://randomuser.me/api/portraits/men/32.jpg',
    badges: ['🚀'],
    lastActive: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    isCurrentUser: true,
  },
  {
    userId: 'user234',
    name: 'Alice Brown',
    rank: 4,
    score: 850,
    avatarUrl: 'https://randomuser.me/api/portraits/women/45.jpg',
    lastActive: new Date(Date.now() - 120 * 60 * 1000).toISOString(),
  },
  {
    userId: 'user345',
    name: 'Mike Wilson',
    rank: 5,
    score: 820,
    avatarUrl: 'https://randomuser.me/api/portraits/men/65.jpg',
    lastActive: new Date(Date.now() - 240 * 60 * 1000).toISOString(),
  },
];

export default function CandlestickQuizPage() {
  const [quizComplete, setQuizComplete] = useState(false);
  const [quizResults, setQuizResults] = useState(null);

  const handleQuizComplete = (results: any) => {
    setQuizComplete(true);
    setQuizResults(results);
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold text-center mb-8">Candlestick Pattern Quiz</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main quiz area */}
        <div className="lg:col-span-2">
          <CandlestickQuiz
            assessmentId={MOCK_ASSESSMENT_ID}
            authToken={MOCK_AUTH_TOKEN}
            initialQuestions={MOCK_QUESTIONS}
            showExplanations={true}
            showTimer={true}
            showFeedback={true}
            allowSkip={false}
            onComplete={handleQuizComplete}
          />
        </div>

        {/* Sidebar with leaderboard */}
        <div>
          <Leaderboard
            boardId={MOCK_ASSESSMENT_ID}
            title="Top Performers"
            authToken={MOCK_AUTH_TOKEN}
            initialEntries={MOCK_LEADERBOARD}
            maxEntries={10}
            showAvatar={true}
            showRankChange={true}
            showBadges={true}
            showLastActive={true}
            currentUserId={MOCK_USER_ID}
          />

          <div className="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">About This Quiz</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              This interactive quiz tests your knowledge of candlestick patterns and chart analysis. 
              Identify the patterns correctly to earn points and climb the leaderboard.
            </p>
            <h3 className="text-lg font-medium mb-2">Instructions:</h3>
            <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-1">
              <li>Analyze each chart carefully</li>
              <li>Select all patterns you can identify</li>
              <li>Submit your answer before the timer runs out</li>
              <li>Review feedback to improve your skills</li>
              <li>Complete all questions to see your final score</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 