import { motion, useAnimationFrame } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useRef, useState } from "react";
import SearchBar from "../SearchBar";

// Particle component for background
const Particle = ({ index }: { index: number }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ x: number; y: number }>(() => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
  }));

  useAnimationFrame((t) => {
    const speed = 0.5;
    const angle = (t * 0.0005 + index * 0.5) % (2 * Math.PI);

    setPosition({
      x: position.x + Math.cos(angle) * speed * 0.1,
      y: position.y + Math.sin(angle) * speed * 0.1,
    });
    if (position.x < 0)
      setPosition((p: { x: number; y: number }) => ({ ...p, x: 100 }));
    if (position.x > 100)
      setPosition((p: { x: number; y: number }) => ({ ...p, x: 0 }));
    if (position.y < 0)
      setPosition((p: { x: number; y: number }) => ({ ...p, y: 100 }));
    if (position.y > 100)
      setPosition((p: { x: number; y: number }) => ({ ...p, y: 0 }));
  });

  return (
    <motion.div
      ref={ref}
      className="absolute w-1 h-1 bg-green-500 rounded-full"
      style={{
        left: `${position.x}%`,
        top: `${position.y}%`,
        boxShadow: "0 0 20px 2px rgba(34, 197, 94, 0.2)",
      }}
      initial={{ opacity: 0 }}
      animate={{ opacity: [0.2, 0.5, 0.2] }}
      transition={{
        duration: 3 + index * 0.2,
        repeat: Infinity,
        ease: "linear",
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

  const finesse = "FINE$$E";

  return (
    <div className="min-h-screen bg-[#0a0d14] text-white flex items-center justify-center relative overflow-hidden">
      {/* Dynamic Background */}
      <div className="absolute inset-0 overflow-hidden z-0">
        {/* Particle system */}
        {Array.from({ length: 30 }).map((_, i) => (
          <Particle key={i} index={i} />
        ))}

        {/* Gradient overlays */}
        <div className="absolute inset-0 bg-gradient-to-b from-green-950/30 via-gray-900/50 to-black/80" />

        {/* Glowing circle in the center */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
          <motion.div
            className="w-[600px] h-[600px] rounded-full bg-gradient-to-r from-green-500/5 to-green-600/5 blur-3xl"
            animate={{
              scale: [1, 1.1, 1],
              opacity: [0.3, 0.4, 0.3],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        </div>

        {/* Horizontal lines effect */}
        <div className="absolute inset-0 overflow-hidden opacity-10">
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={i}
              className="h-px w-full bg-gradient-to-r from-transparent via-green-500/30 to-transparent"
              style={{ transform: `translateY(${i * 5}vh)` }}
            />
          ))}
        </div>
      </div>

      <motion.section
        ref={heroRef}
        initial="hidden"
        animate={heroInView ? "visible" : "hidden"}
        className="px-4 py-20 max-w-7xl mx-auto text-center relative z-10"
      >
        <motion.h1 className="text-6xl md:text-8xl font-extrabold mb-10 tracking-wide">
          <span className="inline-flex">
            {[...finesse].map((char, index) => (
              <motion.span
                key={index}
                variants={letterVariants}
                custom={index}
                className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-green-600"
                style={{
                  display: "inline-block",
                  margin: "0 0.05em",
                  textShadow: "0 0 40px rgba(34, 197, 94, 0.5)",
                }}
              >
                {char}
              </motion.span>
            ))}
          </span>
        </motion.h1>

        <motion.div
          variants={fadeInUp}
          className="relative py-5 px-8 rounded-2xl mx-auto max-w-3xl"
        >
          <SearchBar darkMode={true} />
        </motion.div>
      </motion.section>
    </div>
  );
};

export default LandingPage;
