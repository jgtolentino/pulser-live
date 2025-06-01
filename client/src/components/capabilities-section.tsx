import { motion } from "framer-motion";
import { useScrollReveal } from "@/hooks/use-scroll-reveal";

export default function CapabilitiesSection() {
  const { ref: titleRef, isVisible: titleVisible } = useScrollReveal();

  const capabilities = [
    {
      icon: "fas fa-lightbulb",
      title: "Brand Strategy",
      description: "Uncover insights that drive meaningful brand differentiation and sustainable competitive advantage."
    },
    {
      icon: "fas fa-palette",
      title: "Creative Excellence",
      description: "Award-winning creative solutions that break through the noise and create lasting connections."
    },
    {
      icon: "fas fa-chart-line",
      title: "Media Innovation",
      description: "Strategic media planning and buying that maximizes reach and drives measurable results."
    },
    {
      icon: "fas fa-mobile-alt",
      title: "Digital Transformation",
      description: "Future-forward digital experiences that engage audiences across all touchpoints."
    },
    {
      icon: "fas fa-users",
      title: "Experience Design",
      description: "Human-centered design that creates memorable and meaningful brand interactions."
    },
    {
      icon: "fas fa-globe",
      title: "Global Integration",
      description: "Seamless coordination across markets with local relevance and global consistency."
    }
  ];

  return (
    <section id="capabilities" className="py-24 bg-black">
      <div className="container mx-auto px-6">
        <motion.div
          ref={titleRef}
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={titleVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-5xl md:text-6xl font-black mb-6">
            OUR <span className="text-tbwa-red">CAPABILITIES</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            From strategy to execution, we offer comprehensive solutions that drive business transformation
          </p>
        </motion.div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {capabilities.map((capability, index) => {
            const { ref, isVisible } = useScrollReveal();
            
            return (
              <motion.div
                key={index}
                ref={ref}
                className="bg-tbwa-dark p-8 rounded-2xl group cursor-pointer"
                initial={{ opacity: 0, y: 30 }}
                animate={isVisible ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: 1.05, transition: { duration: 0.3 } }}
              >
                <motion.div
                  className="text-tbwa-red text-4xl mb-6"
                  whileHover={{ scale: 1.1 }}
                  transition={{ duration: 0.3 }}
                >
                  <i className={capability.icon}></i>
                </motion.div>
                <h3 className="text-2xl font-bold mb-4">{capability.title}</h3>
                <p className="text-gray-400 leading-relaxed">
                  {capability.description}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
