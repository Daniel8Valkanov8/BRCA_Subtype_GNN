export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'sans-serif'] },
      colors: {
        clinical: {
          50:  '#f0f7ff',
          100: '#e0effe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          900: '#0c1a2e',
        }
      }
    }
  },
  plugins: []
}
