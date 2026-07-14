import type { MetadataRoute } from "next";

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: "https://pyqed.org",
      changeFrequency: "monthly",
      priority: 1,
    },
    {
      url: "https://pyqed.org/examples",
      changeFrequency: "monthly",
      priority: 0.8,
    },
    {
      url: "https://pyqed.org/viewer",
      changeFrequency: "monthly",
      priority: 0.8,
    },
    {
      url: "https://pyqed.org/privacy",
      changeFrequency: "yearly",
      priority: 0.2,
    },
  ];
}
