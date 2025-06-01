import { motion } from "framer-motion";
import { useScrollReveal } from "@/hooks/use-scroll-reveal";

export default function DisruptionSection() {
  const { ref: textRef, isVisible: textVisible } = useScrollReveal();
  const { ref: imageRef, isVisible: imageVisible } = useScrollReveal();

  return (
    <section id="disruption" className="py-24 bg-tbwa-dark">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <motion.div
            ref={textRef}
            initial={{ opacity: 0, x: -50 }}
            animate={textVisible ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-5xl md:text-6xl font-black mb-8">
              DISRUPTION<br />
              <span className="text-tbwa-red">IS OUR</span><br />
              METHODOLOGY
            </h2>
            
            <p className="text-xl text-gray-300 mb-8 leading-relaxed">
              For over 50 years, Disruption® has been our unique strategic methodology. It's how we identify the conventional wisdom that's holding brands back and find new ways forward.
            </p>
            
            <div className="space-y-6">
              {[
                {
                  title: "Challenge Convention",
                  description: "Question everything that's considered normal in your category"
                },
                {
                  title: "Create Vision",
                  description: "Develop a compelling future state that inspires action"
                },
                {
                  title: "Drive Change",
                  description: "Execute breakthrough ideas that move markets"
                }
              ].map((item, index) => (
                <motion.div
                  key={index}
                  className="flex items-start gap-4"
                  initial={{ opacity: 0, y: 20 }}
                  animate={textVisible ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.6, delay: 0.2 + index * 0.1 }}
                >
                  <div className="w-2 h-2 bg-tbwa-red rounded-full mt-3 flex-shrink-0" />
                  <div>
                    <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                    <p className="text-gray-400">{item.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
          
          <motion.div
            ref={imageRef}
            initial={{ opacity: 0, x: 50 }}
            animate={imageVisible ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
            whileHover={{ scale: 1.05 }}
          >
            <img
              src="https://images.unsplash.com/photo-1542744173-8e7e53415bb0?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=600"
              alt="Modern creative office space"
              className="rounded-2xl shadow-2xl transition-transform duration-300"
            />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
