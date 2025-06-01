import { useState } from "react";
import { motion } from "framer-motion";
import { useScrollReveal } from "@/hooks/use-scroll-reveal";
import { useToast } from "@/hooks/use-toast";

export default function ContactSection() {
  const { ref: titleRef, isVisible: titleVisible } = useScrollReveal();
  const { ref: contactRef, isVisible: contactVisible } = useScrollReveal();
  const { ref: formRef, isVisible: formVisible } = useScrollReveal();
  const { toast } = useToast();

  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    company: '',
    projectType: '',
    message: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    // Simulate form submission
    setTimeout(() => {
      toast({
        title: "Message Sent!",
        description: "Thank you for your interest. We'll be in touch soon.",
      });
      setFormData({
        firstName: '',
        lastName: '',
        email: '',
        company: '',
        projectType: '',
        message: ''
      });
      setIsSubmitting(false);
    }, 1000);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const contactInfo = [
    {
      icon: "fas fa-map-marker-alt",
      title: "TBWA London",
      description: "76-80 Whitfield Street, London W1T 4EZ"
    },
    {
      icon: "fas fa-phone",
      title: "+44 20 7573 6666",
      description: "Monday - Friday, 9AM - 6PM"
    },
    {
      icon: "fas fa-envelope",
      title: "hello@tbwa.com",
      description: "For new business inquiries"
    }
  ];

  const socialLinks = [
    { icon: "fab fa-linkedin", href: "#" },
    { icon: "fab fa-twitter", href: "#" },
    { icon: "fab fa-instagram", href: "#" },
    { icon: "fab fa-youtube", href: "#" }
  ];

  return (
    <section id="contact" className="py-24 bg-tbwa-dark">
      <div className="container mx-auto px-6">
        <motion.div
          ref={titleRef}
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={titleVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-5xl md:text-6xl font-black mb-6">
            LET'S <span className="text-tbwa-red">DISRUPT</span><br />
            TOGETHER
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Ready to challenge conventions and create extraordinary brand experiences? Let's start the conversation.
          </p>
        </motion.div>
        
        <div className="grid lg:grid-cols-2 gap-16">
          <motion.div
            ref={contactRef}
            initial={{ opacity: 0, x: -50 }}
            animate={contactVisible ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
          >
            <div className="space-y-8">
              <div>
                <h3 className="text-2xl font-bold mb-4">Get in Touch</h3>
                <div className="space-y-4">
                  {contactInfo.map((info, index) => (
                    <motion.div
                      key={index}
                      className="flex items-center gap-4"
                      initial={{ opacity: 0, y: 20 }}
                      animate={contactVisible ? { opacity: 1, y: 0 } : {}}
                      transition={{ duration: 0.6, delay: index * 0.1 }}
                    >
                      <div className="w-12 h-12 bg-tbwa-red rounded-full flex items-center justify-center">
                        <i className={`${info.icon} text-white`}></i>
                      </div>
                      <div>
                        <p className="font-semibold">{info.title}</p>
                        <p className="text-gray-400">{info.description}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-2xl font-bold mb-4">Follow Us</h3>
                <div className="flex gap-4">
                  {socialLinks.map((social, index) => (
                    <motion.a
                      key={index}
                      href={social.href}
                      className="w-12 h-12 bg-gray-800 hover:bg-tbwa-red rounded-full flex items-center justify-center transition-colors duration-300"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <i className={`${social.icon} text-white`}></i>
                    </motion.a>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
          
          <motion.div
            ref={formRef}
            initial={{ opacity: 0, x: 50 }}
            animate={formVisible ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
          >
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-semibold mb-2">First Name *</label>
                  <input
                    type="text"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    required
                    className="w-full bg-black border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-tbwa-red focus:outline-none transition-colors duration-300"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-2">Last Name *</label>
                  <input
                    type="text"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    required
                    className="w-full bg-black border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-tbwa-red focus:outline-none transition-colors duration-300"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-semibold mb-2">Email *</label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full bg-black border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-tbwa-red focus:outline-none transition-colors duration-300"
                />
              </div>
              
              <div>
                <label className="block text-sm font-semibold mb-2">Company</label>
                <input
                  type="text"
                  name="company"
                  value={formData.company}
                  onChange={handleChange}
                  className="w-full bg-black border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-tbwa-red focus:outline-none transition-colors duration-300"
                />
              </div>
              
              <div>
                <label className="block text-sm font-semibold mb-2">Project Type</label>
                <select
                  name="projectType"
                  value={formData.projectType}
                  onChange={handleChange}
                  className="w-full bg-black border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-tbwa-red focus:outline-none transition-colors duration-300"
                >
                  <option value="">Select a project type</option>
                  <option value="brand-strategy">Brand Strategy</option>
                  <option value="creative-campaign">Creative Campaign</option>
                  <option value="digital-transformation">Digital Transformation</option>
                  <option value="media-planning">Media Planning</option>
                  <option value="other">Other</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-semibold mb-2">Message *</label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  required
                  rows={5}
                  placeholder="Tell us about your project..."
                  className="w-full bg-black border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-tbwa-red focus:outline-none transition-colors duration-300 resize-none"
                />
              </div>
              
              <motion.button
                type="submit"
                disabled={isSubmitting}
                className="w-full bg-tbwa-red hover:bg-red-700 text-white py-4 text-lg font-semibold transition-all duration-300 rounded-lg disabled:opacity-50"
                whileHover={{ scale: isSubmitting ? 1 : 1.02 }}
                whileTap={{ scale: isSubmitting ? 1 : 0.98 }}
              >
                {isSubmitting ? 'SENDING...' : 'SEND MESSAGE'}
              </motion.button>
            </form>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
