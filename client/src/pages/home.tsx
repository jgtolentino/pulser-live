import Navigation from "@/components/navigation";
import HeroSection from "@/components/hero-section";
import DisruptionSection from "@/components/disruption-section";
import CapabilitiesSection from "@/components/capabilities-section";
import FeaturedWorkSection from "@/components/featured-work-section";
import InsightsSection from "@/components/insights-section";
import ContactSection from "@/components/contact-section";
import Footer from "@/components/footer";

export default function Home() {
  return (
    <div className="bg-tbwa-darker text-white font-inter overflow-x-hidden">
      <Navigation />
      <HeroSection />
      <DisruptionSection />
      <CapabilitiesSection />
      <FeaturedWorkSection />
      <InsightsSection />
      <ContactSection />
      <Footer />
    </div>
  );
}
