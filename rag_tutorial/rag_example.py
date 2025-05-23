import requests

# Dữ liệu listing giả lập (có thể thay thế từ API thật hoặc file CSV/JSON sau)
LISTING_DATA = {
    "listing_id": "L12345",
    "category": "Xe máy cũ",
    "listing_price": "₫12,000,000",
    "position": 18,
    "age_days": 6,
    "views_last_7_days": 120,
    "views_previous_7_days": 90,
    "total_leads": 8,
    "leads_last_7_days": 5,
    "image_count": 3,
    "image_quality_score": 0.7,
    "seller_monthly_spend": "₫250,000",
    "previous_services_used": "Bump, Sticky Ad",
    "bump_success_rate": "70%",
    "sticky_success_rate": "80%",
    "special_display_success_rate": "65%",
    "category_avg_views": 25,
    "category_avg_leads": 1.8,
    "category_avg_position": 10,
    "avg_time_to_sell": "7 days"
}

# Prompt template
PROMPT_TEMPLATE = """
You are a marketing optimization assistant for a classifieds platform.

Available Premium Services:
1. Bump (₫50,000): Moves listing to the top of the category. Position decreases as new listings are posted.
2. Sticky Ad (₫100,000): Pins the listing in positions 1-5 of the category for 30 minutes.
3. Special Display Ad (₫150,000): Enhanced visual presentation with multiple images and 2x display space.

Listing information:
- Listing ID: {listing_id}
- Category: {category}
- Price: {listing_price}
- Current position: {position}
- Age: {age_days} days
- Views (last 7 days): {views_last_7_days}
- Views (previous 7 days): {views_previous_7_days}
- Total leads: {total_leads}
- Leads (last 7 days): {leads_last_7_days}
- Number of images: {image_count}
- Image quality score: {image_quality_score}

Seller history:
- Monthly spend: {seller_monthly_spend}
- Used services: {previous_services_used}
- Success rate with Bump: {bump_success_rate}
- Success rate with Sticky Ad: {sticky_success_rate}
- Success rate with Special Display: {special_display_success_rate}

Category averages:
- Avg views/day: {category_avg_views}
- Avg leads/day: {category_avg_leads}
- Avg position for sold items: {category_avg_position}
- Avg time to sell: {avg_time_to_sell}

Question:
Which premium service (Bump, Sticky Ad, or Special Display Ad) would you recommend for this listing, and why? Consider cost-effectiveness, past performance, and category trends.

Answer in a concise and actionable manner.
"""

def build_prompt(data: dict) -> str:
    return PROMPT_TEMPLATE.format(**data)

def call_ollama(prompt: str, model: str = "gemma:7b") -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Lỗi khi gọi Ollama: {e}"

if __name__ == "__main__":
    prompt = build_prompt(LISTING_DATA)
    print("=== Prompt gửi đến mô hình ===\n", prompt)
    print("\n=== Gợi ý từ Gemma ===\n")
    print(call_ollama(prompt))
    print("\n=== Kết thúc ===")