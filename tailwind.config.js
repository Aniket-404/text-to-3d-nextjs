/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#121212",
        surface: "#1E1E1E",
        primary: "#00D1FF",
        accent: "#FF007A",
        "text-primary": "#E0E0E0",
        "text-secondary": "#AAAAAA",
      },
      fontFamily: {
        sans: ["Inter", "Poppins", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 15px rgba(0, 209, 255, 0.5)",
        "glow-accent": "0 0 15px rgba(255, 0, 122, 0.5)",
      },
      animation: {
        "spin-slow": "spin 3s linear infinite",
        pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        shimmer: "shimmer 2s infinite linear",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "card-gradient": "linear-gradient(145deg, rgba(30, 30, 30, 0.6), rgba(30, 30, 30, 0.8))",
        "input-gradient": "linear-gradient(145deg, rgba(18, 18, 18, 0.8), rgba(30, 30, 30, 0.6))",
        "text-gradient": "linear-gradient(90deg, #00D1FF, #FF007A)",
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
};
