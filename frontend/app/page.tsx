"use client";

import { useState } from "react";

type Message = { role: "user" | "ai"; content: string };

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = (input?: string) => {
    const q = input || query;
    if (!q) return;

    setLoading(true);

    const userMessage: Message = { role: "user", content: q };
    const aiMessage: Message = { role: "ai", content: "" };

    setMessages((prev) => [...prev, userMessage, aiMessage]);
    setResults([]);

    const eventSource = new EventSource(
      `http://127.0.0.1:8000/stream?query=${encodeURIComponent(q)}`
    );

    let currentText = "";

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.token) {
        currentText = data.token;

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "ai",
            content: currentText,
          };
          return updated;
        });
      }

      if (data.results) {
        setResults(data.results);
        setLoading(false);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setLoading(false);
    };

    setQuery("");
  };

  return (
    <div className="h-screen flex bg-gradient-to-b from-black to-gray-900 text-white">

      {/* LEFT - CHAT */}
      <div className="w-1/2 flex flex-col border-r border-gray-800">

        {/* Header */}
        <div className="p-4 text-xl font-bold border-b border-gray-800 flex justify-between">
          🍽️ FoodGPT
          <span className="text-sm text-gray-400">AI Food Discovery</span>
        </div>

        {/* Chat messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-md px-4 py-3 rounded-2xl shadow ${msg.role === "user"
                    ? "bg-blue-600"
                    : "bg-gray-800"
                  }`}
              >
                {msg.role === "ai" ? (
                  <div className="space-y-2">
                    {msg.content
                      .split("\n")
                      .filter((line) => line.trim().match(/^\d+\./))
                      .map((line, idx) => {
                        const clean = line.replace(/^\d+\.\s*/, "");
                        return (
                          <div key={idx} className="bg-gray-700 p-2 rounded text-sm">
                            <span className="text-blue-400 font-semibold">
                              #{idx + 1}
                            </span>{" "}
                            {clean}
                          </div>
                        );
                      })}
                  </div>
                ) : (
                  msg.content
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="text-gray-400 text-sm animate-pulse">
              AI is thinking...
            </div>
          )}
        </div>

        {/* Suggestions */}
        <div className="px-4 pb-2 flex gap-2 flex-wrap">
          {["cheap chicken", "date night", "veg food", "under ₹300"].map((s) => (
            <button
              key={s}
              onClick={() => handleSend(s)}
              className="bg-gray-800 px-3 py-1 rounded-full text-sm hover:bg-gray-700"
            >
              {s}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-800 flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Try: cheap chicken under ₹300"
            className="flex-1 p-3 rounded-xl bg-gray-900 outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={() => handleSend()}
            className="bg-blue-600 px-5 py-2 rounded-xl hover:bg-blue-500"
          >
            Send
          </button>
        </div>
      </div>

      {/* RIGHT - RESULTS */}
      <div className="w-1/2 p-6 overflow-y-auto">

        <h2 className="text-lg font-semibold mb-4">Top Matches</h2>

        {results.length === 0 && !loading && (
          <div className="text-gray-500 text-sm">
            Your recommendations will appear here 👀
          </div>
        )}

        <div className="space-y-4">
          {results.map((r, i) => (
            <div
              key={i}
              className="bg-gray-900 p-4 rounded-xl border border-gray-800 hover:border-blue-500 transition"
            >
              <div className="text-lg font-semibold">{r.name}</div>

              <div className="text-sm text-gray-400 mt-1">
                📍 {r.location}
              </div>

              <div className="text-sm mt-2">
                🍜 {r.cuisines}
              </div>

              <div className="text-sm mt-1">
                ⭐ {r.rate}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}