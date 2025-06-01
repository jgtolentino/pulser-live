import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export default function Navigation() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 100);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setIsMenuOpen(false);
    }
  };

  return (
    <nav className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ${
      isScrolled ? 'bg-black/95' : 'bg-black/90'
    } backdrop-blur-sm`}>
      <div className="container mx-auto px-6 py-4 flex justify-between items-center">
        <motion.div 
          className="text-2xl font-bold cursor-pointer"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          onClick={() => scrollToSection('home')}
        >
          <span className="text-white">TBWA</span>
          <span className="text-tbwa-red">\\</span>
          <span className="text-white">LONDON</span>
        </motion.div>
        
        <motion.div 
          className="hidden md:flex space-x-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {[
            { label: 'Home', id: 'home' },
            { label: 'Disruption', id: 'disruption' },
            { label: 'Capabilities', id: 'capabilities' },
            { label: 'Work', id: 'work' },
            { label: 'Contact', id: 'contact' }
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className="hover:text-tbwa-red transition-colors duration-300"
            >
              {item.label}
            </button>
          ))}
        </motion.div>
        
        <button
          className="md:hidden text-white"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <i className="fas fa-bars text-xl"></i>
        </button>
      </div>
      
      {/* Mobile Menu */}
      <motion.div
        className={`md:hidden bg-black/95 backdrop-blur-sm ${isMenuOpen ? 'block' : 'hidden'}`}
        initial={{ opacity: 0, height: 0 }}
        animate={{ 
          opacity: isMenuOpen ? 1 : 0, 
          height: isMenuOpen ? 'auto' : 0 
        }}
        transition={{ duration: 0.3 }}
      >
        <div className="px-6 py-4 space-y-4">
          {[
            { label: 'Home', id: 'home' },
            { label: 'Disruption', id: 'disruption' },
            { label: 'Capabilities', id: 'capabilities' },
            { label: 'Work', id: 'work' },
            { label: 'Contact', id: 'contact' }
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className="block hover:text-tbwa-red transition-colors duration-300"
            >
              {item.label}
            </button>
          ))}
        </div>
      </motion.div>
    </nav>
  );
}
