import { motion } from "framer-motion";

export default function Footer() {
  const footerSections = [
    {
      title: "Services",
      links: [
        { label: "Brand Strategy", href: "#" },
        { label: "Creative", href: "#" },
        { label: "Media", href: "#" },
        { label: "Digital", href: "#" }
      ]
    },
    {
      title: "Company",
      links: [
        { label: "About", href: "#" },
        { label: "Careers", href: "#" },
        { label: "News", href: "#" },
        { label: "Contact", href: "#" }
      ]
    },
    {
      title: "Legal",
      links: [
        { label: "Privacy Policy", href: "#" },
        { label: "Terms of Use", href: "#" },
        { label: "Cookies", href: "#" }
      ]
    }
  ];

  return (
    <footer className="bg-black py-12 border-t border-gray-800">
      <div className="container mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <div className="text-2xl font-bold mb-4">
              <span className="text-white">TBWA</span>
              <span className="text-tbwa-red">\\</span>
              <span className="text-white">LONDON</span>
            </div>
            <p className="text-gray-400 text-sm">
              Disruption® is our unique methodology for challenging conventional wisdom and creating breakthrough brand experiences.
            </p>
          </motion.div>
          
          {footerSections.map((section, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: (index + 1) * 0.1 }}
              viewport={{ once: true }}
            >
              <h4 className="font-semibold mb-4">{section.title}</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                {section.links.map((link, linkIndex) => (
                  <li key={linkIndex}>
                    <motion.a
                      href={link.href}
                      className="hover:text-white transition-colors duration-300"
                      whileHover={{ x: 5 }}
                      transition={{ duration: 0.2 }}
                    >
                      {link.label}
                    </motion.a>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
        
        <motion.div
          className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
        >
          <p>&copy; 2024 TBWA London. All rights reserved. Disruption® is a registered trademark of TBWA Worldwide.</p>
        </motion.div>
      </div>
    </footer>
  );
}
