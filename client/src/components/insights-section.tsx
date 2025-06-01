import { motion } from "framer-motion";
import { useScrollReveal } from "@/hooks/use-scroll-reveal";

export default function InsightsSection() {
  const { ref: imageRef, isVisible: imageVisible } = useScrollReveal();
  const { ref: textRef, isVisible: textVisible } = useScrollReveal();

  const insights = [
    {
      title: "The Art of Meaningful Disruption",
      description: "How brands can create positive change while driving business growth"
    },
    {
      title: "Future of Brand Experience",
      description: "Emerging technologies reshaping how consumers interact with brands"
    },
    {
      title: "Cultural Intelligence",
      description: "Understanding global trends that influence local brand behavior"
    }
  ];

  return (
    <section className="py-24 bg-black">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <motion.div
            ref={imageRef}
            initial={{ opacity: 0, x: -50 }}
            animate={imageVisible ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
            whileHover={{ scale: 1.02 }}
          >
            <img
              src="https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=600"
              alt="Abstract geometric design patterns"
              className="rounded-2xl shadow-2xl transition-transform duration-300"
            />
          </motion.div>
          
          <motion.div
            ref={textRef}
            initial={{ opacity: 0, x: 50 }}
            animate={textVisible ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-5xl md:text-6xl font-black mb-8">
              INSIGHTS &<br />
              <span className="text-tbwa-red">CULTURE</span>
            </h2>
            
            <p className="text-xl text-gray-300 mb-8 leading-relaxed">
              Stay ahead of the curve with our latest thinking on brand strategy, cultural trends, and the future of marketing.
            </p>
            
            <div className="space-y-6 mb-8">
              {insights.map((insight, index) => (
                <motion.div
                  key={index}
                  className="border-l-4 border-tbwa-red pl-6"
                  initial={{ opacity: 0, x: -20 }}
                  animate={textVisible ? { opacity: 1, x: 0 } : {}}
                  transition={{ duration: 0.6, delay: 0.2 + index * 0.1 }}
                  whileHover={{ x: 10 }}
                >
                  <h3 className="text-xl font-semibold mb-2">{insight.title}</h3>
                  <p className="text-gray-400">{insight.description}</p>
                </motion.div>
              ))}
            </div>
            
            <motion.button
              className="border-2 border-tbwa-red hover:bg-tbwa-red text-tbwa-red hover:text-white px-8 py-4 text-lg font-semibold transition-all duration-300 rounded-full"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              EXPLORE INSIGHTS
            </motion.button>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
