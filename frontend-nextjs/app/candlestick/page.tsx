'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import axios from 'axios';
import { motion } from 'framer-motion';
import CandlestickChart from '../components/CandlestickChart';

// Mock data for candlestick patterns (used for descriptions only)
const CANDLESTICK_PATTERNS = [
  { id: 1, name: 'Doji', description: 'A candlestick pattern with a very small body where opening and closing prices are almost equal.' },
  { id: 2, name: 'Hammer', description: 'A bullish reversal pattern with a small body and a long lower shadow.' },
  { id: 3, name: 'Shooting Star', description: 'A bearish reversal pattern with a small body and a long upper shadow.' },
  { id: 4, name: 'Bullish Engulfing', description: 'A bullish reversal pattern where a small bearish candle is followed by a larger bullish candle that engulfs it.' },
  { id: 5, name: 'Bearish Engulfing', description: 'A bearish reversal pattern where a small bullish candle is followed by a larger bearish candle that engulfs it.' },
  { id: 6, name: 'Morning Star', description: 'A bullish reversal pattern consisting of three candles: a large bearish candle, a small-bodied candle, and a large bullish candle.' },
  { id: 7, name: 'Evening Star', description: 'A bearish reversal pattern consisting of three candles: a large bullish candle, a small-bodied candle, and a large bearish candle.' },
  { id: 8, name: 'Long-Legged Doji', description: 'A doji with long upper and lower shadows, indicating significant indecision in the market.' },
  { id: 9, name: 'Dragonfly Doji', description: 'A doji with a long lower shadow and little to no upper shadow, often signaling a bullish reversal.' },
  { id: 10, name: 'Gravestone Doji', description: 'A doji with a long upper shadow and little to no lower shadow, often signaling a bearish reversal.' },
  { id: 11, name: 'Ladder Bottom', description: 'A bullish reversal pattern consisting of a series of bearish candles followed by a strong bullish candle.' },
];

// API Configuration
const API_URL = 'http://localhost:8000/api/assessments/candlestick';
const AUTH_TOKEN = 'test-token'; // For testing purposes

export default function CandlestickAssessment() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<any>(null);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [score, setScore] = useState(0);
  const [assessmentComplete, setAssessmentComplete] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<any | null>(null);
  const [questionStartTime, setQuestionStartTime] = useState<number | null>(null);
  const [timeLeft, setTimeLeft] = useState<number | null>(null);
  const [sessionSummary, setSessionSummary] = useState<any | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [totalAnswers, setTotalAnswers] = useState(0);

  // Start a new assessment session on component mount
  useEffect(() => {
    startNewSession();
  }, []);
  
  // Timer effect for counting down the time limit
  useEffect(() => {
    if (!currentQuestion || timeLeft === null) return;
    
    const timer = setInterval(() => {
      setTimeLeft((prevTime) => {
        if (prevTime === null || prevTime <= 0) {
          clearInterval(timer);
          // Auto-submit when time runs out
          if (!feedback) {
            handleSubmitAnswer();
          }
          return 0;
        }
        return prevTime - 1;
      });
    }, 1000);
    
    return () => clearInterval(timer);
  }, [currentQuestion, timeLeft, feedback]);
  
  // Start a new assessment session
  const startNewSession = async () => {
    setLoading(true);
    setError(null);
    setScore(0);
    setAssessmentComplete(false);
    setFeedback(null);
    setSessionSummary(null);
    setCurrentQuestion(null); // Initialize to null to prevent map error
    
    try {
      const response = await axios.post(`${API_URL}/start`, {
        difficulty: 0.5,
        total_questions: 5
      }, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`
        }
      });
      
      if (response.data && response.data.session_id) {
        setSessionId(response.data.session_id);
        
        // Format and set the current question data
        const questionData = {
          ...response.data,
          question_id: response.data.question?.id || response.data.question_id || "",
          question_number: 1,
          total_questions: response.data.total_questions || 5,
          time_limit_seconds: response.data.time_limit_seconds || 30,
          options: response.data.question?.options || [],
          image_data: response.data.question?.chart_data || []
        };
        
        setCurrentQuestion(questionData);
        setTimeLeft(questionData.time_limit_seconds);
        setQuestionStartTime(Date.now());
      } else {
        console.error('Invalid response format:', response.data);
        setError('Invalid data received from server');
        handleMockFallback(); // Use mock data as fallback
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error starting assessment:', err);
      setError('Failed to start the assessment. Using mock data instead.');
      setLoading(false);
      
      // Use mock data as fallback
      handleMockFallback();
    }
  };
  
  // Handle option selection
  const handleOptionSelect = (option: string) => {
    if (feedback || !currentQuestion) return; // Prevent selection if no question or already answered
    setSelectedOption(option);
  };
  
  // Submit the selected answer
  const handleSubmitAnswer = async () => {
    if (!sessionId || !currentQuestion || !selectedOption) {
      setError('Please select an option before submitting');
      return;
    }
    
    setLoading(true);
    const responseTimeMs = questionStartTime ? Date.now() - questionStartTime : 5000;
    
    try {
      const response = await axios.post(`${API_URL}/submit_answer`, {
        session_id: sessionId,
        question_id: currentQuestion.question_id,
        selected_option: selectedOption,
        response_time_ms: responseTimeMs
      }, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`
        }
      });
      
      // Update score
      if (response.data && response.data.score !== undefined) {
        setScore(response.data.score);
      }
      
      // Show feedback
      if (response.data && response.data.explanation) {
        setFeedback(response.data.explanation);
      } else {
        // If no explanation, create a simple feedback object
        setFeedback({ is_correct: false, pattern_name: "Unknown" });
      }
      
      // Check if assessment is complete
      if (response.data && response.data.assessment_complete) {
        setAssessmentComplete(true);
        if (sessionId) {
          fetchSessionDetails(sessionId);
        }
      } else {
        // Prepare for next question
        if (response.data && response.data.next_question) {
          setCurrentQuestion(response.data.next_question);
        } else {
          // If next_question is missing, try to use a fallback
          console.log("Next question data not available in response. Using fallback.");
          handleMockFallback();
        }
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error submitting answer:', err);
      setError('Failed to submit your answer. Please try again.');
      setLoading(false);
    }
  };
  
  // Fetch session details
  const fetchSessionDetails = async (sid: string) => {
    try {
      const response = await axios.get(`${API_URL}/session/${sid}`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`
        }
      });
      setSessionSummary(response.data);
    } catch (err) {
      console.error('Error fetching session details:', err);
    }
  };
  
  // Handle moving to the next question
  const handleNextQuestion = () => {
    if (!currentQuestion) return;
    
    setSelectedOption(null);
    setFeedback(null);
    setQuestionStartTime(Date.now());
    setTimeLeft(currentQuestion.time_limit_seconds || 30); // Default to 30 seconds if not specified
  };
  
  // Mock function to handle API errors - in a real app, this would use the actual API
  const handleMockFallback = () => {
    // This is a fallback for when the real API isn't available or for development
    if (!currentQuestion) {
      const mockQuestion = {
        question_id: "mock-id-" + Math.random().toString(36).substr(2, 9),
        question_number: 1,
        total_questions: 5,
        question_text: "What candlestick pattern is shown in this chart?",
        options: [
          "Doji",
          "Hammer",
          "Bullish Engulfing",
          "Evening Star"
        ],
        image_data: [
          { time: '2023-01-01', open: 100, high: 105, low: 95, close: 102, volume: 1000 },
          { time: '2023-01-02', open: 102, high: 110, low: 100, close: 109, volume: 1500 },
          { time: '2023-01-03', open: 109, high: 115, low: 107, close: 110, volume: 1200 },
          { time: '2023-01-04', open: 110, high: 112, low: 106, close: 107, volume: 1000 },
          { time: '2023-01-05', open: 107, high: 107, low: 100, close: 102, volume: 1300 }
        ],
        time_limit_seconds: 30,
        difficulty: 0.5,
        session_id: "mock-session-" + Math.random().toString(36).substr(2, 9)
      };
      
      setSessionId(mockQuestion.session_id);
      setCurrentQuestion(mockQuestion);
      setTimeLeft(mockQuestion.time_limit_seconds);
      setQuestionStartTime(Date.now());
      setLoading(false);
      return;
    }
    
    if (selectedOption && !feedback) {
      const correctAnswer = "Doji"; // For demo purposes
      const isCorrect = selectedOption === correctAnswer;
      
      const mockFeedback = {
        is_correct: isCorrect,
        pattern_name: correctAnswer,
        user_level: "beginner",
        components: {
          pattern_definition: "A doji is a candlestick pattern with a very small body where opening and closing prices are almost equal.",
          visual_characteristics: "The candle has a small or nonexistent body with upper and lower shadows.",
          market_psychology: "Represents indecision in the market, with neither buyers nor sellers gaining control.",
          trading_implications: "Often signals a potential trend reversal, especially after a strong uptrend or downtrend.",
          common_mistakes: "Traders sometimes confuse doji with small-bodied candles or misinterpret the significance based on position."
        },
        historical_examples: [
          "Example: USD/JPY, January 15, 2020 - A doji formed at the top of an uptrend, signaling the start of a reversal.",
          "Example: Apple Inc., March 23, 2021 - A doji at support level preceded a significant rally."
        ]
      };
      
      setFeedback(mockFeedback);
      setScore(prev => isCorrect ? prev + 10 : prev);
      
      // If this is the last question, complete the assessment
      if (currentQuestion.question_number >= 5) {
        setAssessmentComplete(true);
        setSessionSummary({
          session_id: sessionId,
          total_questions: 5,
          completed_questions: 5,
          correct_answers: isCorrect ? 3 : 2,
          avg_response_time: 4500,
          score: isCorrect ? 30 : 20,
          started_at: Math.floor(Date.now() / 1000) - 300,
          completed_at: Math.floor(Date.now() / 1000),
          user_id: "mock-user",
          accuracy: isCorrect ? 0.6 : 0.4
        });
      }
      return;
    }
    
    if (feedback && !assessmentComplete) {
      // Mock a new question
      const qNum = (currentQuestion.question_number || 0) + 1;
      const mockNextQuestion = {
        question_id: "mock-id-" + Math.random().toString(36).substr(2, 9),
        question_number: qNum,
        total_questions: 5,
        question_text: "What candlestick pattern is shown in this chart?",
        options: qNum % 2 === 0 ? 
          ["Hammer", "Shooting Star", "Morning Star", "Doji"] :
          ["Bearish Engulfing", "Bullish Engulfing", "Evening Star", "Long-Legged Doji"],
        image_data: qNum % 2 === 0 ? 
          [
            { time: '2023-01-01', open: 110, high: 115, low: 95, close: 100, volume: 1000 },
            { time: '2023-01-02', open: 100, high: 110, low: 90, close: 95, volume: 1500 },
            { time: '2023-01-03', open: 95, high: 100, low: 85, close: 90, volume: 1200 },
            { time: '2023-01-04', open: 90, high: 95, low: 80, close: 89, volume: 1000 },
            { time: '2023-01-05', open: 89, high: 95, low: 89, close: 95, volume: 1300 }
          ] :
          [
            { time: '2023-01-01', open: 95, high: 100, low: 90, close: 97, volume: 1000 },
            { time: '2023-01-02', open: 97, high: 105, low: 95, close: 102, volume: 1500 },
            { time: '2023-01-03', open: 102, high: 110, low: 100, close: 108, volume: 1200 },
            { time: '2023-01-04', open: 108, high: 115, low: 105, close: 112, volume: 1000 },
            { time: '2023-01-05', open: 112, high: 120, low: 110, close: 118, volume: 1300 }
          ],
        time_limit_seconds: 30,
        difficulty: 0.5,
        session_id: sessionId
      };
      
      setCurrentQuestion(mockNextQuestion);
      setTimeLeft(mockNextQuestion.time_limit_seconds);
      setQuestionStartTime(Date.now());
      setSelectedOption(null);
      setFeedback(null);
    }
  };
  
  const incrementCorrectAnswers = useCallback(() => {
    setCorrectAnswers((prev: number) => prev + 1);
    setTotalAnswers((prev: number) => prev + 1);
  }, []);

  const incrementTotalAnswers = useCallback(() => {
    setTotalAnswers((prev: number) => prev + 1);
  }, []);
  
  // Loading state
  if (loading && !currentQuestion) {
    return (
      <div className="max-w-4xl mx-auto p-6 flex flex-col items-center justify-center min-h-screen">
        <h1 className="text-3xl font-bold mb-8 text-center">Candlestick Pattern Assessment</h1>
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700"></div>
        </div>
        <p className="mt-4 text-gray-600">Loading assessment questions...</p>
      </div>
    );
  }
  
  // Error state
  if (error && !currentQuestion) {
    return (
      <div className="max-w-4xl mx-auto p-6 flex flex-col items-center justify-center min-h-screen">
        <h1 className="text-3xl font-bold mb-8 text-center">Candlestick Pattern Assessment</h1>
        <div className="bg-red-100 text-red-800 p-4 rounded-lg mb-6">
          {error}
        </div>
        <button
          onClick={startNewSession}
          className="py-2 px-6 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }
  
  // Assessment completion screen
  if (assessmentComplete && sessionSummary) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-8 text-center">Candlestick Pattern Assessment</h1>
        
        <div className="bg-gray-100 p-8 rounded-lg text-center">
          <h2 className="text-2xl font-bold mb-4">Assessment Complete!</h2>
          
          <div className="text-5xl font-bold mb-6">
            {sessionSummary.correct_answers} / {sessionSummary.total_questions}
          </div>
          
          <div className="mb-6">
            <p className="text-lg font-medium">Score: {sessionSummary.score} points</p>
            <p className="text-md text-gray-600">Accuracy: {Math.round(sessionSummary.accuracy * 100)}%</p>
            <p className="text-md text-gray-600">Average response time: {Math.round(sessionSummary.avg_response_time / 1000)} seconds</p>
          </div>
          
          <p className="text-lg mb-8">
            {sessionSummary.accuracy >= 0.8 
              ? 'Excellent! You have a strong understanding of candlestick patterns.'
              : sessionSummary.accuracy >= 0.6 
                ? 'Good job! Keep practicing to improve your pattern recognition skills.'
                : 'Keep learning! Recognizing candlestick patterns takes practice.'}
          </p>
          
          <div className="flex justify-center space-x-4">
            <button
              onClick={startNewSession}
              className="py-2 px-6 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors"
            >
              Try Again
            </button>
            
            <Link 
              href="/"
              className="py-2 px-6 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium transition-colors"
            >
              Back to Home
            </Link>
          </div>
        </div>
      </div>
    );
  }
  
  // Active assessment view
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-blue-900 to-gray-900">
      {/* Navigation */}
      <nav className="bg-white/10 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Link href="/" className="text-2xl font-bold text-white hover:text-blue-200 transition-colors">
                TradeIQ
              </Link>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className="text-blue-200 hover:text-white transition-colors"
              >
                {darkMode ? '☀️' : '🌙'}
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold text-white mb-4"
          >
            Candlestick Pattern Assessment
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-blue-200 text-lg"
          >
            Identify the candlestick patterns and predict market movements
          </motion.p>
        </div>

        {/* Chart Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white/5 backdrop-blur-sm border border-blue-500/30 rounded-2xl p-6 mb-8"
        >
          <div className="h-[400px]">
            <CandlestickChart data={currentQuestion?.image_data || []} darkMode={darkMode} />
          </div>
        </motion.div>

        {/* Question Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white/5 backdrop-blur-sm border border-blue-500/30 rounded-2xl p-6"
        >
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-white mb-4">
              {currentQuestion ? 
                `Question ${currentQuestion.question_number || 1} of ${currentQuestion.total_questions || 5}` : 
                "Loading question..."
              }
            </h3>
            <p className="text-blue-200">What candlestick pattern is forming in the highlighted area?</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {currentQuestion && currentQuestion.options && currentQuestion.options.map((option: string) => (
              <button
                key={option}
                onClick={() => handleOptionSelect(option)}
                disabled={!!feedback}
                className={`p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-blue-500/30 hover:border-blue-500/50 text-white text-left transition-all duration-300 hover:transform hover:-translate-y-1 ${
                  selectedOption === option
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'opacity-50'
                }`}
              >
                {option}
              </button>
            ))}
          </div>

          <div className="mt-8 flex justify-between items-center">
            <button
              className="px-6 py-3 rounded-lg bg-white/10 text-blue-200 hover:bg-white/20 transition-all duration-300"
              disabled={!currentQuestion || currentQuestion.question_number === 1}
              onClick={() => setCurrentQuestion((prev: number) => Math.max(1, prev - 1))}
            >
              Previous
            </button>
            <div className="text-blue-200">
              {currentQuestion ? 
                `${currentQuestion.question_number || 1} / ${currentQuestion.total_questions || 5}` :
                "Loading..."
              }
            </div>
            <button
              className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-emerald-500 text-white hover:from-blue-600 hover:to-emerald-600 transition-all duration-300"
              disabled={!currentQuestion}
              onClick={() => currentQuestion && setCurrentQuestion((prev: number) => Math.min(currentQuestion.total_questions, prev + 1))}
            >
              Next
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="bg-gray-50 rounded-lg p-4 text-center">
      <div className="text-2xl mb-2">{icon}</div>
      <div className="text-sm text-gray-600">{label}</div>
      <div className="font-semibold text-gray-900">{value}</div>
    </div>
  );
}

function FeedbackSection({ title, content }: { title: string; content: string }) {
  return (
    <div>
      <h4 className="font-medium text-gray-900 mb-1">{title}</h4>
      <p className="text-gray-600 text-sm">{content}</p>
    </div>
  );
} 