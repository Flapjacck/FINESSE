import { useState } from "react";
import { FaSearch } from "react-icons/fa";
import { useNavigate } from "react-router-dom";

interface SearchBarProps {
  darkMode?: boolean;
  className?: string;
}

const SearchBar = ({ darkMode = true, className = "" }: SearchBarProps) => {
  const [isFocused, setIsFocused] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const navigate = useNavigate();

  const handleSearch = () => {
    if (searchValue) {
      navigate(`/stock/${searchValue}`);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className={`relative ${className}`}>
      <div
        className={`relative backdrop-blur-xl ${
          darkMode ? "bg-gray-900/30" : "bg-white"
        } rounded-2xl p-2 border-2 transition-colors duration-300 ${
          isFocused
            ? "border-green-500/50"
            : darkMode
            ? "border-green-500/10"
            : "border-gray-200"
        }`}
      >
        <div className="flex items-center space-x-4">
          <div className="relative flex-grow">
            <div
              className={`absolute left-4 top-1/2 -translate-y-1/2 ${
                darkMode ? "text-green-500/50" : "text-green-600"
              }`}
            >
              $
            </div>
            <input
              type="text"
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value.toUpperCase())}
              onKeyDown={handleKeyPress}
              placeholder="Enter a ticker..."
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              className={`w-full py-3 pl-8 pr-4 rounded-xl ${
                darkMode
                  ? "bg-gray-800/50 text-white placeholder-gray-400"
                  : "bg-gray-50 text-gray-900 placeholder-gray-500"
              } focus:outline-none text-lg tracking-wider`}
              style={{ caretColor: "#22c55e" }}
            />
          </div>

          <button
            onClick={handleSearch}
            className={`p-4 ${
              darkMode ? "bg-green-500" : "bg-green-600"
            } rounded-xl hover:brightness-110 focus:outline-none relative group transition-all duration-200`}
          >
            <FaSearch className="text-white text-xl" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default SearchBar;
