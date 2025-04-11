'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Types for feedback messages
export interface FeedbackMessage {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning' | 'hint';
  message: string;
  detail?: string;
  timestamp: string | Date;
  isRead?: boolean;
  isPinned?: boolean;
  source?: string;
}

interface FeedbackPanelProps {
  messages: FeedbackMessage[];
  onMarkAsRead?: (id: string) => void;
  onPinMessage?: (id: string, isPinned: boolean) => void;
  onClearAll?: () => void;
  onDismissMessage?: (id: string) => void;
  title?: string;
  maxHeight?: string | number;
  showControls?: boolean;
  autoScroll?: boolean;
  showTimestamp?: boolean;
  className?: string;
}

export default function FeedbackPanel({
  messages,
  onMarkAsRead,
  onPinMessage,
  onClearAll,
  onDismissMessage,
  title = 'Feedback',
  maxHeight = '300px',
  showControls = true,
  autoScroll = true,
  showTimestamp = true,
  className = '',
}: FeedbackPanelProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [pinnedMessages, setPinnedMessages] = useState<FeedbackMessage[]>([]);
  const [regularMessages, setRegularMessages] = useState<FeedbackMessage[]>([]);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [newMessageCount, setNewMessageCount] = useState(0);

  // Split messages into pinned and regular
  useEffect(() => {
    const pinned = messages.filter(msg => msg.isPinned);
    const regular = messages.filter(msg => !msg.isPinned);
    
    setPinnedMessages(pinned);
    setRegularMessages(regular);
    
    // Calculate unread messages
    const unreadCount = messages.filter(msg => !msg.isRead).length;
    setNewMessageCount(unreadCount);
  }, [messages]);

  // Auto-scroll to the newest message
  useEffect(() => {
    if (autoScroll && messagesEndRef.current && !isCollapsed) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [regularMessages, isCollapsed, autoScroll]);

  // Format timestamp
  const formatTimestamp = (timestamp: string | Date) => {
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Get icon by message type
  const getIconByType = (type: FeedbackMessage['type']) => {
    switch (type) {
      case 'success':
        return (
          <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'error':
        return (
          <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      case 'hint':
        return (
          <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5 text-gray-500" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        );
    }
  };
  
  // Handle message dismissal
  const handleDismiss = (id: string) => {
    if (onDismissMessage) {
      onDismissMessage(id);
    }
  };
  
  // Handle message pin/unpin
  const handlePin = (id: string, currentPinState: boolean = false) => {
    if (onPinMessage) {
      onPinMessage(id, !currentPinState);
    }
  };
  
  // Handle mark as read
  const handleMarkAsRead = (id: string) => {
    if (onMarkAsRead) {
      onMarkAsRead(id);
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden ${className}`}>
      {/* Header with controls */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            {title}
          </h3>
          {newMessageCount > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100 text-xs font-medium rounded-full">
              {newMessageCount} new
            </span>
          )}
        </div>
        
        <div className="flex space-x-2">
          {showControls && (
            <>
              <button
                onClick={() => onClearAll && onClearAll()}
                className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                Clear All
              </button>
              <button
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 focus:outline-none"
              >
                {isCollapsed ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                  </svg>
                )}
              </button>
            </>
          )}
        </div>
      </div>
      
      {/* Message container */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div
              className="overflow-y-auto p-3 space-y-3"
              style={{ maxHeight }}
            >
              {/* Pinned messages section */}
              {pinnedMessages.length > 0 && (
                <div className="mb-4">
                  <div className="text-xs uppercase text-gray-500 dark:text-gray-400 mb-2 font-semibold tracking-wider">
                    Pinned Messages
                  </div>
                  {pinnedMessages.map((message) => (
                    <FeedbackMessageItem
                      key={message.id}
                      message={message}
                      onDismiss={handleDismiss}
                      onPin={handlePin}
                      onMarkAsRead={handleMarkAsRead}
                      showTimestamp={showTimestamp}
                      formatTimestamp={formatTimestamp}
                      getIconByType={getIconByType}
                    />
                  ))}
                </div>
              )}
              
              {/* Regular messages */}
              {regularMessages.map((message) => (
                <FeedbackMessageItem
                  key={message.id}
                  message={message}
                  onDismiss={handleDismiss}
                  onPin={handlePin}
                  onMarkAsRead={handleMarkAsRead}
                  showTimestamp={showTimestamp}
                  formatTimestamp={formatTimestamp}
                  getIconByType={getIconByType}
                />
              ))}
              
              {/* Auto-scroll reference */}
              <div ref={messagesEndRef} />
              
              {/* Empty state */}
              {messages.length === 0 && (
                <div className="text-center py-6 text-gray-500 dark:text-gray-400">
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <p className="mt-2 text-sm">No feedback messages yet</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Individual message component
function FeedbackMessageItem({
  message,
  onDismiss,
  onPin,
  onMarkAsRead,
  showTimestamp,
  formatTimestamp,
  getIconByType,
}: {
  message: FeedbackMessage;
  onDismiss: (id: string) => void;
  onPin: (id: string, isPinned: boolean) => void;
  onMarkAsRead: (id: string) => void;
  showTimestamp: boolean;
  formatTimestamp: (timestamp: string | Date) => string;
  getIconByType: (type: FeedbackMessage['type']) => React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3 }}
      className={`
        relative p-3 rounded-lg border 
        ${message.isRead ? 'bg-white dark:bg-gray-800' : 'bg-blue-50 dark:bg-blue-900/20'} 
        ${message.type === 'success' && 'border-green-200 dark:border-green-800'}
        ${message.type === 'error' && 'border-red-200 dark:border-red-800'}
        ${message.type === 'warning' && 'border-yellow-200 dark:border-yellow-800'}
        ${message.type === 'info' && 'border-gray-200 dark:border-gray-700'}
        ${message.type === 'hint' && 'border-blue-200 dark:border-blue-800'}
      `}
      onClick={() => !message.isRead && onMarkAsRead(message.id)}
    >
      <div className="flex">
        <div className="flex-shrink-0 mr-3">
          {getIconByType(message.type)}
        </div>
        
        <div className="flex-1 min-w-0">
          <p className={`text-sm ${!message.isRead ? 'font-medium' : ''} text-gray-900 dark:text-gray-100`}>
            {message.message}
          </p>
          
          {message.detail && (
            <p className="mt-1 text-xs text-gray-600 dark:text-gray-300">
              {message.detail}
            </p>
          )}
          
          {showTimestamp && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {formatTimestamp(message.timestamp)}
              {message.source && ` • ${message.source}`}
            </p>
          )}
        </div>
        
        <div className="ml-2 flex-shrink-0 flex flex-col space-y-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onPin(message.id, !!message.isPinned);
            }}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 focus:outline-none"
            aria-label={message.isPinned ? "Unpin message" : "Pin message"}
          >
            <svg className={`w-4 h-4 ${message.isPinned ? 'text-blue-500 fill-current' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
            </svg>
          </button>
          
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDismiss(message.id);
            }}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 focus:outline-none"
            aria-label="Dismiss message"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
      
      {!message.isRead && (
        <div className="absolute top-0 right-0 w-2 h-2 m-1 rounded-full bg-blue-500" />
      )}
    </motion.div>
  );
} 