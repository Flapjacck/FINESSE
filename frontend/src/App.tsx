import "./App.css";
import LandingPage from "./components/pages/landing";
import { StockTest } from "./components/StockTest";

function App() {
  return (
    <div className="app min-h-screen bg-gradient-to-b from-green-50 via-green-100 to-white">
      <LandingPage />
    </div>
  );
}

export default App;
