import { createRoot } from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
  Navigate,
} from "react-router-dom";
import App from "./App.tsx";
import JoinPage from "./pages/JoinPage.tsx";
import CreateCallPage from "./pages/CreateCallPage.tsx";
import JoinByIdPage from "./pages/JoinByIdPage.tsx";

import "./globals.css";
import { Toaster } from "sonner";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        index: true, // This makes it the default child route for "/"
        element: <Navigate to="/join" replace />,
      },
      {
        path: "join",
        element: <JoinPage />,
      },
      {
        path: "join/:id",
        element: <JoinByIdPage />,
      },
      {
        path: "create",
        element: <CreateCallPage />,
      },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <>
    <RouterProvider router={router} />
    <Toaster richColors />
  </>
);
