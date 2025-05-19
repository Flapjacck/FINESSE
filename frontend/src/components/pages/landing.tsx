import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";

interface BackgroundElementProps {
  x: number;
  y: number;
  delay: number;
  size: string;
  opacity: number;
  duration: number;
}

// Custom background element component
const BackgroundElement = ({
  x,
  y,
  delay,
  size,
  opacity,
  duration,
}: BackgroundElementProps) => {
  return (
    <motion.div
      className="absolute rounded-full bg-gradient-to-br from-green-300 to-green-500 blur-xl"
      style={{
        top: `${y}%`,
        left: `${x}%`,
        width: size,
        height: size,
      }}
      initial={{ opacity: 0, scale: 0 }}
      animate={{
        opacity,
        scale: [1, 1.2, 1],
        x: [0, -10, 10, -5, 0],
        y: [0, 5, -5, 0],
      }}
      transition={{
        delay,
        duration,
        repeat: Infinity,
        repeatType: "reverse",
      }}
    />
  );
};

const LandingPage = () => {
  const [heroRef, heroInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });
  const letterVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.12,
        duration: 0.6,
        ease: [0.22, 1, 0.36, 1],
      },
    }),
  };
  const fadeInUp = {
    hidden: { opacity: 0, y: 40 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 1,
        ease: [0.22, 1, 0.36, 1],
        delay: 0.8,
      },
    },
  };
  const finesse = "FINESSE";
  // Generate background elements data
  const backgroundElements = [
    { x: 15, y: 20, size: "180px", opacity: 0.15, delay: 0, duration: 15 },
    { x: 85, y: 30, size: "220px", opacity: 0.12, delay: 2, duration: 18 },
    { x: 70, y: 65, size: "240px", opacity: 0.14, delay: 4, duration: 20 },
    { x: 10, y: 80, size: "200px", opacity: 0.13, delay: 1, duration: 17 },
    { x: 50, y: 12, size: "260px", opacity: 0.08, delay: 3.5, duration: 22 },
    { x: 30, y: 60, size: "160px", opacity: 0.15, delay: 5, duration: 16 },
    { x: 80, y: 75, size: "180px", opacity: 0.11, delay: 2.5, duration: 19 },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 via-green-100 to-white text-slate-900 flex items-center justify-center relative overflow-hidden">
      {/* Dynamic background elements */}
      <div className="absolute inset-0 overflow-hidden z-0">
        {backgroundElements.map((el, index) => (
          <BackgroundElement
            key={index}
            x={el.x}
            y={el.y}
            size={el.size}
            opacity={el.opacity}
            delay={el.delay}
            duration={el.duration}
          />
        ))}

        {/* Mesh overlay for depth */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0)_0%,rgba(255,255,255,0.8)_70%)]" />
      </div>

      <motion.section
        ref={heroRef}
        initial="hidden"
        animate={heroInView ? "visible" : "hidden"}
        className="px-4 py-20 max-w-7xl mx-auto text-center relative z-10"
      >
        {" "}
        <motion.h1 className="text-5xl md:text-7xl lg:text-8xl font-bold mb-10 tracking-normal flex justify-center">
          <span className="inline-flex overflow-visible">
            {[...finesse].map((char, index) => (
              <motion.span
                key={index}
                variants={letterVariants}
                custom={index}
                className="text-green-600 relative z-10"
                style={{
                  display: "inline-block",
                  padding: "0 0.02em", // Add slight padding to prevent cutting off
                  margin: "0 0.01em", // Add slight margin between letters
                  textShadow: "0 4px 16px rgba(74, 222, 128, 0.4)",
                  background: "linear-gradient(to bottom, #22c55e, #16a34a)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}
              >
                {char}
              </motion.span>
            ))}
          </span>
        </motion.h1>{" "}
        <motion.div
          className="relative backdrop-blur-md bg-white/50 py-5 px-8 rounded-xl shadow-md mx-auto max-w-3xl"
          variants={fadeInUp}
        >
          <motion.p className="text-xl md:text-2xl text-slate-800 max-w-2xl mx-auto leading-relaxed font-medium">
            Streamline your financial management with powerful tools and
            AI-driven insights
          </motion.p>
          <motion.div
            className="absolute -z-10 w-full h-2 bg-gradient-to-r from-green-200 via-green-300 to-green-200 rounded-full left-0 bottom-0 opacity-70"
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ delay: 1.5, duration: 1.8, ease: "easeInOut" }}
          />
        </motion.div>{" "}
        <motion.div
          className="w-full max-w-md h-2 bg-gradient-to-r from-transparent via-green-400 to-transparent mx-auto mt-16 rounded-full shadow-green-500/50 shadow-sm"
          initial={{ scaleX: 0, opacity: 0, transformOrigin: "center" }}
          animate={{ scaleX: 1, opacity: 0.8 }}
          transition={{ delay: 2.2, duration: 1.8 }}
        />
      </motion.section>
    </div>
  );
};

export default LandingPage;
