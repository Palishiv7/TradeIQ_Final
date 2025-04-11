'use client';

import React, { useState, useEffect, useCallback } from 'react';
import CandlestickChart from './CandlestickChart';
import FeedbackPanel, { FeedbackMessage } from './FeedbackPanel';
import { useAssessmentWebSocket } from '../utils/websocket';
import { motion, AnimatePresence } from 'framer-motion';

// Types for the quiz
export interface CandlestickPattern {
  id: string;
  name: string;
  description: string;
  reliability: 'high' | 'medium' | 'low';
  bullish: boolean;
  bearish: boolean;
}

export interface CandlestickQuizQuestion {
  id: string;
  chartData: any[]; // CandlestickData from lightweight-charts
  patterns: CandlestickPattern[];
  correctPatternIds: string[];
  explanation?: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  timeLimit?: number; // in seconds
}

interface CandlestickQuizProps {
  assessmentId: string;
  authToken: string;
  initialQuestions?: CandlestickQuizQuestion[];
  showExplanations?: boolean;
  showTimer?: boolean;
  showFeedback?: boolean;
  allowSkip?: boolean;
  onComplete?: (results: any) => void;
  className?: string;
}

export default function CandlestickQuiz({
  assessmentId,
  authToken,
  initialQuestions = [],
  showExplanations = true,
  showTimer = true,
  showFeedback = true,
  allowSkip = false,
  onComplete,
  className = '',
}: CandlestickQuizProps) {
  // State
  const [questions, setQuestions] = useState<CandlestickQuizQuestion[]>(initialQuestions);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedPatternIds, setSelectedPatternIds] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAnswered, setIsAnswered] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackMessage[]>([]);
  const [remainingTime, setRemainingTime] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [results, setResults] = useState<any>(null);

  // Current question
  const currentQuestion = questions[currentQuestionIndex] || null;
  
  // Connect to WebSocket for real-time feedback
  const { 
    status: wsStatus, 
    lastMessage, 
    sendMessage 
  } = useAssessmentWebSocket(assessmentId, authToken);
  
  // Initialize quiz
  useEffect(() => {
    if (currentQuestion && currentQuestion.timeLimit && showTimer) {
      setRemainingTime(currentQuestion.timeLimit);
    }
  }, [currentQuestion, showTimer]);
  
  // Handle timer countdown
  useEffect(() => {
    if (remainingTime === null || isAnswered || !showTimer) return;
    
    const timer = setInterval(() => {
      setRemainingTime(prev => {
        if (prev === null || prev <= 0) {
          clearInterval(timer);
          // Auto-submit when time runs out
          if (!isAnswered) {
            handleSubmit();
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    
    return () => clearInterval(timer);
  }, [remainingTime, isAnswered, showTimer]);
  
  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const { type, data } = lastMessage;
      
      if (type === 'assessment_feedback') {
        // Add feedback message
        const newFeedback: FeedbackMessage = {
          id: Date.now().toString(),
          type: data.feedbackType || 'info',
          message: data.message,
          detail: data.detail,
          timestamp: new Date(),
          isRead: false,
        };
        
        setFeedback(prev => [newFeedback, ...prev]);
      } else if (type === 'score_update') {
        // Update score
        setScore(data.score);
      } else if (type === 'next_question') {
        // Load next question from server
        const newQuestion: CandlestickQuizQuestion = data.question;
        
        if (newQuestion) {
          setQuestions(prev => [...prev, newQuestion]);
          // Move to the new question if we're at the end
          if (currentQuestionIndex === questions.length - 1) {
            goToNextQuestion();
          }
        }
      } else if (type === 'assessment_complete') {
        // Handle assessment completion
        setIsComplete(true);
        setResults(data.results);
        
        if (onComplete) {
          onComplete(data.results);
        }
      }
    }
  }, [lastMessage, currentQuestionIndex, questions]);
  
  // Toggle pattern selection
  const togglePatternSelection = (patternId: string) => {
    if (isAnswered) return;
    
    setSelectedPatternIds(prev => {
      if (prev.includes(patternId)) {
        return prev.filter(id => id !== patternId);
      } else {
        return [...prev, patternId];
      }
    });
  };
  
  // Submit answer
  const handleSubmit = useCallback(() => {
    if (!currentQuestion || isSubmitting || isAnswered) return;
    
    setIsSubmitting(true);
    
    // Send answer to server via WebSocket
    sendMessage({
      type: 'submit_answer',
      questionId: currentQuestion.id,
      selectedPatternIds,
      timeSpent: currentQuestion.timeLimit ? (currentQuestion.timeLimit - (remainingTime || 0)) : null,
    });
    
    // Mark as answered
    setIsAnswered(true);
    setRemainingTime(null);
    
    // Local feedback about correctness
    const correctIds = currentQuestion.correctPatternIds;
    const isCorrect = 
      selectedPatternIds.length === correctIds.length && 
      selectedPatternIds.every(id => correctIds.includes(id));
    
    // Add feedback message for correct/incorrect
    const feedbackMessage: FeedbackMessage = {
      id: Date.now().toString(),
      type: isCorrect ? 'success' : 'error',
      message: isCorrect ? 'Correct!' : 'Not quite right',
      detail: showExplanations ? currentQuestion.explanation : undefined,
      timestamp: new Date(),
      isRead: false,
    };
    
    setFeedback(prev => [feedbackMessage, ...prev]);
    setIsSubmitting(false);
  }, [currentQuestion, isSubmitting, isAnswered, selectedPatternIds, remainingTime, sendMessage, showExplanations]);
  
  // Go to next question
  const goToNextQuestion = useCallback(() => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
      setSelectedPatternIds([]);
      setIsAnswered(false);
      
      // Reset timer if next question has a time limit
      const nextQuestion = questions[currentQuestionIndex + 1];
      if (nextQuestion && nextQuestion.timeLimit && showTimer) {
        setRemainingTime(nextQuestion.timeLimit);
      } else {
        setRemainingTime(null);
      }
      
      // Scroll to top
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      // No more questions, request completion if not already completed
      if (!isComplete) {
        sendMessage({
          type: 'request_completion',
        });
      }
    }
  }, [currentQuestionIndex, questions, showTimer, isComplete, sendMessage]);
  
  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Mark feedback as read
  const handleMarkFeedbackAsRead = (id: string) => {
    setFeedback(prev => 
      prev.map(msg => 
        msg.id === id ? { ...msg, isRead: true } : msg
      )
    );
  };
  
  // Pin feedback message
  const handlePinFeedback = (id: string, isPinned: boolean) => {
    setFeedback(prev => 
      prev.map(msg => 
        msg.id === id ? { ...msg, isPinned } : msg
      )
    );
  };
  
  // Clear all feedback
  const handleClearFeedback = () => {
    setFeedback([]);
  };
  
  // Dismiss feedback message
  const handleDismissFeedback = (id: string) => {
    setFeedback(prev => prev.filter(msg => msg.id !== id));
  };

  // If no questions are available
  if (questions.length === 0) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${className}`}>
        <div className="text-center">
          <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-white">
            No questions available
          </h3>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            Please try again later or contact support.
          </p>
        </div>
      </div>
    );
  }

  // If assessment is complete
  if (isComplete && results) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden ${className}`}>
        <div className="p-6">
          <div className="text-center mb-6">
            <svg className="mx-auto h-16 w-16 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2 className="mt-4 text-2xl font-bold text-gray-900 dark:text-white">
              Assessment Complete!
            </h2>
            <p className="mt-2 text-gray-600 dark:text-gray-300">
              You've completed the candlestick pattern assessment.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400">Final Score</p>
              <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">{results.score}</p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400">Correct Answers</p>
              <p className="text-3xl font-bold text-green-600 dark:text-green-400">{results.correctCount}</p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400">Completion Time</p>
              <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">{results.completionTime}</p>
            </div>
          </div>
          
          {results.feedback && (
            <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h3 className="text-lg font-medium text-blue-800 dark:text-blue-200 mb-2">Feedback</h3>
              <p className="text-blue-700 dark:text-blue-300">{results.feedback}</p>
            </div>
          )}
          
          <div className="flex justify-center">
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Try Another Assessment
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden ${className}`}>
      {/* Quiz header */}
      <div className="bg-gray-50 dark:bg-gray-700 px-4 py-3 flex items-center justify-between border-b border-gray-200 dark:border-gray-600">
        <div>
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            Candlestick Pattern Quiz
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Question {currentQuestionIndex + 1} of {questions.length}
          </p>
        </div>
        
        <div className="flex items-center">
          {showTimer && remainingTime !== null && (
            <div className={`text-lg font-mono mr-4 ${
              remainingTime < 10 ? 'text-red-600 dark:text-red-400' : 'text-gray-700 dark:text-gray-300'
            }`}>
              {formatTime(remainingTime)}
            </div>
          )}
          
          <div className="flex items-center">
            <span className="text-gray-600 dark:text-gray-300 mr-2">Score:</span>
            <span className="font-medium">{score}</span>
          </div>
        </div>
      </div>
      
      {/* Main content */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
        {/* Chart section */}
        <div className="md:col-span-2">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg overflow-hidden">
            <CandlestickChart
              data={currentQuestion.chartData}
              height={400}
              darkMode={false}
              showVolume={true}
              showGrid={true}
              showCrosshair={true}
              showTooltip={true}
              className="w-full"
            />
          </div>
          
          {/* Question prompt */}
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              Identify the candlestick pattern(s)
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Select all patterns you can identify in the chart above.
            </p>
          </div>
        </div>
        
        {/* Answer section */}
        <div>
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
            <div className="p-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Available Patterns
              </h3>
              
              <div className="space-y-2">
                {currentQuestion.patterns.map((pattern) => (
                  <div 
                    key={pattern.id}
                    onClick={() => togglePatternSelection(pattern.id)}
                    className={`
                      p-3 rounded-lg border-2 cursor-pointer transition-colors
                      ${selectedPatternIds.includes(pattern.id) 
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'}
                      ${isAnswered && currentQuestion.correctPatternIds.includes(pattern.id)
                        ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                        : ''}
                      ${isAnswered && selectedPatternIds.includes(pattern.id) && !currentQuestion.correctPatternIds.includes(pattern.id)
                        ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                        : ''}
                    `}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">
                          {pattern.name}
                        </p>
                        <div className="flex items-center mt-1 space-x-2">
                          {pattern.bullish && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                              Bullish
                            </span>
                          )}
                          {pattern.bearish && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                              Bearish
                            </span>
                          )}
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
                            {pattern.reliability.charAt(0).toUpperCase() + pattern.reliability.slice(1)} reliability
                          </span>
                        </div>
                      </div>
                      
                      <div className="h-6 w-6 rounded-full border-2 flex items-center justify-center">
                        {selectedPatternIds.includes(pattern.id) && (
                          <div className="h-3 w-3 rounded-full bg-blue-500 dark:bg-blue-400" />
                        )}
                      </div>
                    </div>
                    
                    {isAnswered && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        transition={{ duration: 0.3 }}
                        className="mt-2 text-sm"
                      >
                        {currentQuestion.correctPatternIds.includes(pattern.id) ? (
                          <p className="text-green-600 dark:text-green-400">
                            This is a correct pattern.
                          </p>
                        ) : selectedPatternIds.includes(pattern.id) ? (
                          <p className="text-red-600 dark:text-red-400">
                            This is not the right pattern.
                          </p>
                        ) : null}
                      </motion.div>
                    )}
                  </div>
                ))}
              </div>
              
              <div className="mt-6 flex space-x-3">
                <button
                  onClick={handleSubmit}
                  disabled={isSubmitting || isAnswered || selectedPatternIds.length === 0}
                  className={`
                    flex-1 px-4 py-2 rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-offset-2
                    ${isSubmitting || isAnswered || selectedPatternIds.length === 0
                      ? 'bg-gray-300 text-gray-500 dark:bg-gray-700 dark:text-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
                    }
                  `}
                >
                  {isSubmitting ? 'Submitting...' : isAnswered ? 'Submitted' : 'Submit Answer'}
                </button>
                
                {(isAnswered || allowSkip) && (
                  <button
                    onClick={goToNextQuestion}
                    disabled={isSubmitting || currentQuestionIndex >= questions.length - 1 && !isAnswered}
                    className={`
                      flex-1 px-4 py-2 rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-offset-2
                      ${isSubmitting || (currentQuestionIndex >= questions.length - 1 && !isAnswered)
                        ? 'bg-gray-300 text-gray-500 dark:bg-gray-700 dark:text-gray-400 cursor-not-allowed'
                        : 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500'
                      }
                    `}
                  >
                    {currentQuestionIndex >= questions.length - 1 ? 'Complete Quiz' : 'Next Question'}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Feedback panel */}
      {showFeedback && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <FeedbackPanel
            messages={feedback}
            onMarkAsRead={handleMarkFeedbackAsRead}
            onPinMessage={handlePinFeedback}
            onClearAll={handleClearFeedback}
            onDismissMessage={handleDismissFeedback}
            title="Quiz Feedback"
            maxHeight="250px"
            autoScroll={true}
          />
        </div>
      )}
      
      {/* WebSocket status indicator */}
      <div className="p-2 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700 flex justify-end items-center text-xs text-gray-500 dark:text-gray-400">
        <span className={`h-2 w-2 rounded-full mr-1 ${
          wsStatus === 'open' 
            ? 'bg-green-500' 
            : wsStatus === 'connecting' || wsStatus === 'reconnecting'
            ? 'bg-yellow-500'
            : 'bg-red-500'
        }`} />
        {wsStatus === 'open' 
          ? 'Connected' 
          : wsStatus === 'connecting'
          ? 'Connecting...'
          : wsStatus === 'reconnecting'
          ? 'Reconnecting...'
          : 'Disconnected'}
      </div>
    </div>
  );
} 