import { motion } from "framer-motion";
import { useScrollReveal } from "@/hooks/use-scroll-reveal";

export default function FeaturedWorkSection() {
  const { ref: titleRef, isVisible: titleVisible } = useScrollReveal();

  const featuredWork = [
    {
      image: "https://images.unsplash.com/photo-1559136555-9303baea8ebd?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=500",
      tag: "GLOBAL CAMPAIGN",
      title: "Disrupting Automotive",
      description: "How we helped a traditional automotive brand challenge the status quo and lead the electric revolution.",
      alt: "Business presentation in conference room"
    },
    {
      image: "https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=500",
      tag: "BRAND TRANSFORMATION",
      title: "Redefining Luxury",
      description: "A comprehensive brand transformation that challenged luxury conventions and created new market dynamics.",
      alt: "Creative team brainstorming session"
    },
    {
      image: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=500",
      tag: "DIGITAL INNOVATION",
      title: "Future of Retail",
      description: "Pioneering digital experiences that blur the lines between physical and virtual commerce.",
      alt: "Abstract geometric design elements"
    },
    {
      image: "https://images.unsplash.com/photo-1586717791821-3f44a563fa4c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=500",
      tag: "SOCIAL IMPACT",
      title: "Purpose-Driven Change",
      description: "Campaigns that drive meaningful social change while building stronger brand connections.",
      alt: "Creative workspace with design tools"
    }
  ];

  return (
    <section id="work" className="py-24 bg-tbwa-dark">
      <div className="container mx-auto px-6">
        <motion.div
          ref={titleRef}
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={titleVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-5xl md:text-6xl font-black mb-6">
            FEATURED <span className="text-tbwa-red">WORK</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Breakthrough campaigns that have redefined categories and created cultural movements
          </p>
        </motion.div>
        
        <div className="grid lg:grid-cols-2 gap-12">
          {featuredWork.map((work, index) => {
            const { ref, isVisible } = useScrollReveal();
            
            return (
              <motion.div
                key={index}
                ref={ref}
                className="group cursor-pointer"
                initial={{ opacity: 0, y: 30 }}
                animate={isVisible ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: index * 0.2 }}
              >
                <motion.div
                  className="relative overflow-hidden rounded-2xl mb-6"
                  whileHover={{ scale: 1.02 }}
                  transition={{ duration: 0.3 }}
                >
                  <motion.img
                    src={work.image}
                    alt={work.alt}
                    className="w-full h-80 object-cover transition-transform duration-500"
                    whileHover={{ scale: 1.1 }}
                  />
                  <div className="absolute inset-0 bg-black/40 group-hover:bg-black/20 transition-colors duration-300" />
                  <div className="absolute top-6 left-6">
                    <span className="bg-tbwa-red text-white px-4 py-2 text-sm font-semibold rounded-full">
                      {work.tag}
                    </span>
                  </div>
                </motion.div>
                <motion.h3
                  className="text-3xl font-bold mb-4 group-hover:text-tbwa-red transition-colors duration-300"
                  whileHover={{ x: 10 }}
                  transition={{ duration: 0.3 }}
                >
                  {work.title}
                </motion.h3>
                <p className="text-gray-400 text-lg leading-relaxed mb-6">
                  {work.description}
                </p>
                <motion.a
                  href="#"
                  className="text-tbwa-red font-semibold hover:underline inline-flex items-center gap-2"
                  whileHover={{ x: 5 }}
                  transition={{ duration: 0.3 }}
                >
                  VIEW CASE STUDY <i className="fas fa-arrow-right"></i>
                </motion.a>
              </motion.div>
            );
          })}
        </div>
        
        <motion.div
          className="text-center mt-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <motion.button
            className="bg-tbwa-red hover:bg-red-700 text-white px-12 py-4 text-lg font-semibold transition-all duration-300 rounded-full"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            VIEW ALL WORK
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}
