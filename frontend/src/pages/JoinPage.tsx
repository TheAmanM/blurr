import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Label } from "../components/ui/label";
import Background from "../components/Background";

export default function JoinPage() {
  return (
    <Background>
      <Card className="w-full max-w-md mx-4">
        <CardHeader>
          <CardTitle className="text-2xl lg:text-3xl">Join a Meeting</CardTitle>
          <CardDescription className="text-muted-foreground lg:text-lg">
            Enter a meeting code to get started.
          </CardDescription>
        </CardHeader>
        <CardContent className="mt-4">
          <Label className="mb-2 text-muted-foreground">Meeting Code</Label>
          <Input placeholder="••••••••••••••••••••" />
        </CardContent>
        <CardFooter>
          <Button className="w-full -mt-2">Join</Button>
        </CardFooter>
      </Card>
    </Background>
  );
}
