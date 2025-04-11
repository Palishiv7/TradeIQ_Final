'use client';

import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-blue-900 to-gray-900">
      {/* Navigation */}
      <nav className="bg-white/10 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <span className="text-2xl font-bold text-white">TradeIQ</span>
            </div>
            <div className="flex items-center space-x-4">
              <Link 
                href="#" 
                className="text-blue-200 hover:text-white transition-colors"
              >
                About
              </Link>
              <Link 
                href="#" 
                className="text-blue-200 hover:text-white transition-colors"
              >
                Contact
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden min-h-screen flex items-center">
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center opacity-20 [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
        
        <div className="relative w-full">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center"
            >
              <h1 className="text-6xl md:text-7xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-emerald-400 to-blue-400 animate-gradient-x">
                Master Trading with AI
              </h1>
              <p className="text-xl md:text-2xl text-blue-100 max-w-3xl mx-auto mb-12 leading-relaxed">
                Enhance your trading skills through our intelligent assessment platform. 
                Get real-time feedback and personalized insights powered by advanced AI.
              </p>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.8 }}
              >
                <Link 
                  href="#assessments"
                  className="inline-flex items-center px-8 py-4 rounded-full bg-gradient-to-r from-blue-500 via-emerald-500 to-blue-500 text-white font-semibold text-lg hover:from-blue-600 hover:via-emerald-600 hover:to-blue-600 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-1 animate-gradient-x"
                  onClick={(e) => {
                    e.preventDefault();
                    document.getElementById('assessments')?.scrollIntoView({ behavior: 'smooth' });
                  }}
                >
                  Start Learning
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </Link>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Assessment Section */}
      <section id="assessments" className="min-h-screen py-24 relative scroll-mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8 }}
            className="text-center mb-20"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Choose Your Assessment Path
            </h2>
            <p className="text-blue-200 text-xl max-w-2xl mx-auto leading-relaxed">
              Select from our specialized assessments designed to enhance different aspects of your trading expertise
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-12">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              <AssessmentCard 
                title="Candlestick Pattern Assessment"
                description="Master the art of reading candlestick patterns. Learn to identify key formations and predict market movements with confidence."
                icon="📊"
                link="/candlestick"
                color="blue"
                stats={{
                  questions: 20,
                  timeEstimate: "30 min",
                  difficulty: "Intermediate"
                }}
              />
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4, duration: 0.8 }}
            >
              <AssessmentCard 
                title="Market Fundamentals Assessment"
                description="Deep dive into market indicators, economic factors, and technical analysis. Build a strong foundation for informed trading decisions."
                icon="📈"
                link="/fundamentals"
                color="green"
                stats={{
                  questions: 25,
                  timeEstimate: "45 min",
                  difficulty: "Advanced"
                }}
                comingSoon
              />
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              <AssessmentCard 
                title="Market Psychology Assessment"
                description="Understand the psychological aspects of trading. Learn to recognize and overcome common biases that affect trading decisions."
                icon="🧠"
                link="/psychology"
                color="purple"
                stats={{
                  questions: 15,
                  timeEstimate: "25 min",
                  difficulty: "Intermediate"
                }}
                comingSoon
              />
            </motion.div>
          </div>
        </div>
      </section>
      
      {/* Features Section */}
      <section className="py-24 relative">
        <div className="absolute inset-0 bg-blue-900/50" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Why Choose TradeIQ
            </h2>
            <p className="text-blue-200 text-lg max-w-2xl mx-auto">
              Experience the future of trading education with our cutting-edge platform
            </p>
          </motion.div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              <FeatureCard
                icon="🤖"
                title="AI-Powered Learning"
                description="Dynamic questions that adapt to your skill level and learning pace"
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4, duration: 0.8 }}
            >
              <FeatureCard
                icon="⚡"
                title="Real-Time Feedback"
                description="Get instant analysis and detailed explanations for every answer"
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              <FeatureCard
                icon="📈"
                title="Track Progress"
                description="Monitor your improvement with comprehensive analytics and insights"
              />
            </motion.div>
          </div>
        </div>
      </section>

      <footer className="border-t border-blue-800/30 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-blue-200">
            TradeIQ - Revolutionizing Financial Market Education with AI
          </p>
        </div>
      </footer>
    </div>
  );
}

interface AssessmentCardProps {
  title: string;
  description: string;
  icon: string;
  link: string;
  color: 'blue' | 'green' | 'purple' | 'orange';
  stats: {
    questions: number;
    timeEstimate: string;
    difficulty: string;
  };
  comingSoon?: boolean;
}

function AssessmentCard({ 
  title, 
  description, 
  icon, 
  link, 
  color, 
  stats,
  comingSoon = false 
}: AssessmentCardProps) {
  const colorClasses = {
    blue: 'bg-white/5 backdrop-blur-sm border-blue-500/30 hover:border-blue-500/50 hover:shadow-blue-500/20',
    green: 'bg-white/5 backdrop-blur-sm border-emerald-500/30 hover:border-emerald-500/50 hover:shadow-emerald-500/20',
    purple: 'bg-white/5 backdrop-blur-sm border-purple-500/30 hover:border-purple-500/50 hover:shadow-purple-500/20',
    orange: 'bg-white/5 backdrop-blur-sm border-orange-500/30 hover:border-orange-500/50 hover:shadow-orange-500/20',
  };
  
  const iconColorClasses = {
    blue: 'bg-blue-500/10 text-blue-300',
    green: 'bg-emerald-500/10 text-emerald-300',
    purple: 'bg-purple-500/10 text-purple-300',
    orange: 'bg-orange-500/10 text-orange-300',
  };
  
  return (
    <div className={`rounded-2xl border ${colorClasses[color]} p-6 transition-all duration-300 hover:shadow-2xl group hover:-translate-y-2 h-full flex flex-col`}>
      <div className={`w-14 h-14 rounded-xl ${iconColorClasses[color]} flex items-center justify-center text-2xl mb-4 transition-transform group-hover:scale-110 backdrop-blur-sm`}>
        {icon}
      </div>
      
      <h2 className="text-lg font-bold mb-2 text-white">{title}</h2>
      <p className="text-blue-200 mb-6 text-sm leading-relaxed">{description}</p>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-3 mb-6 text-xs mt-auto">
        <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2">
          <p className="text-blue-300 mb-1">Questions</p>
          <p className="font-semibold text-white">{stats.questions}</p>
        </div>
        <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2">
          <p className="text-blue-300 mb-1">Time</p>
          <p className="font-semibold text-white">{stats.timeEstimate}</p>
        </div>
        <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2">
          <p className="text-blue-300 mb-1">Level</p>
          <p className="font-semibold text-white">{stats.difficulty}</p>
        </div>
      </div>
      
      {comingSoon ? (
        <div className="bg-white/5 backdrop-blur-sm text-blue-200 py-2 px-4 rounded-lg text-center text-sm font-medium border border-blue-500/30">
          Coming Soon
        </div>
      ) : (
        <Link 
          href={link}
          className="block py-2 px-4 rounded-lg bg-gradient-to-r from-blue-500 via-emerald-500 to-blue-500 text-white text-center text-sm font-medium transition-all duration-300 hover:shadow-lg hover:from-blue-600 hover:via-emerald-600 hover:to-blue-600 transform hover:-translate-y-0.5 animate-gradient-x"
        >
          Start Assessment
        </Link>
      )}
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 text-center border border-blue-500/30 transition-all duration-300 hover:shadow-xl hover:-translate-y-1">
      <div className="text-4xl mb-4 transform transition-transform hover:scale-110">{icon}</div>
      <h3 className="text-xl font-semibold mb-2 text-white">{title}</h3>
      <p className="text-blue-200">{description}</p>
    </div>
  );
} 